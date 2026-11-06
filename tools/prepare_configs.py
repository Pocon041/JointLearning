"""Write portable matched COD-only, IML-only and Joint configurations."""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path


PACKAGES = ("cod_iml_joint", "cod_iml_evp_official_aligned", "cod_iml_evp_training_v2",
            "cod_iml_evp_full_image", "cod_iml_evp_reference", "cod_iml_evp_joint_v3")
MODEL = dict(enhancement="evp", sharing="all_private", frequency_enabled=True,
             frequency_fusion="sum", attention_lora_rank=8, attention_lora_alpha=8.,
             freeze_backbone=True, checkpoint_blocks=True, reference_img_size=352,
             private_stage_exits=True, exit_rank=4, exit_alpha=4.)
STAGES = {"quick": (6, 352), "expanded": (6, 1153), "full": (30, 1153)}


def source_hashes():
    hashes = {}
    for package in PACKAGES:
        folder = Path(importlib.import_module(package).__file__).parent
        hashes[package] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(folder.glob("*.py"))}
    return hashes


def make_configs(*, stage, pretrained, manifest_root, run_root):
    epochs, steps = STAGES[stage]
    manifests = Path(manifest_root).resolve()
    config = dict(seed=42, device="cuda", batch_size=4, workers=4, amp=True, eval_amp=True,
        epochs=epochs, steps_per_epoch=steps, warmup_epochs=105 / 352,
        min_lr=1e-6, warmup_start_factor=.1, shared_lr=2e-4,
        cod_private_lr=2e-4, iml_private_lr=5e-4, weight_decay=.01,
        max_grad_norm=5., clip_mode="global", cod_weight=1., iml_weight=1.,
        boundary_weight=0., iml_dice_weight=0., evaluate_test=stage == "full",
        cod_metrics=True, auroc_bins=4096, log_interval=25,
        selection_metric={"cod":"smeasure", "iml":"tampered_f1"},
        model_variant="evp_joint_v3", model=copy.deepcopy(MODEL),
        shared_optimizer="task_adamw_sum", archive_epoch_checkpoints=True,
        input_geometry={"cod":dict(max_side=512,resize_mode="fit",bucket_step=64),
                        "iml":dict(max_side=512,resize_mode="native",bucket_step=64)},
        size={"cod":[352,352], "iml":[256,256]},
        spatial_pretrained=str(Path(pretrained).resolve()), frozen_package_sources=source_hashes(),
        casia_evaluation_runner_sha256=hashlib.sha256(
            Path(importlib.import_module("tools.casia_train").__file__).read_bytes()).hexdigest())
    # Expanded already uses the full IML training pool, exactly as the experiment.
    train_names = {"cod":f"train_{stage}.jsonl",
                   "iml":"train_quick.jsonl" if stage == "quick" else "train_full.jsonl"}
    config["data"] = {task:{split:str(manifests / task /
        (train_names[task] if split == "train" else split + ".jsonl"))
        for split in (("train","val","test") if stage == "full" else ("train","val"))}
        for task in ("cod","iml")}
    configs = {}
    for arm in ("cod_only", "iml_only", "joint"):
        item = copy.deepcopy(config)
        item.update(arm=arm, output_dir=str(Path(run_root).resolve() / stage / ("both_" + arm)))
        configs["both_" + arm] = item
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="full")
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("configs/local"))
    args = parser.parse_args()
    configs = make_configs(stage=args.stage, pretrained=args.pretrained,
                           manifest_root=args.manifest_root, run_root=args.run_root)
    targets = [args.output / (name + ".json") for name in configs]
    if any(path.exists() for path in targets):
        parser.error("configuration files already exist; choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        path = args.output / (name + ".json")
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(path.resolve())


if __name__ == "__main__":
    main()
