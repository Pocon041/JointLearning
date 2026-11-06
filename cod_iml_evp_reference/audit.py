"""Disposable real-data CUDA audit; never supplies weights to formal runs."""
import argparse
import gc
import json
import inspect
import re
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from cod_iml_evp_bridge_v2.engine import joint_train_step
from cod_iml_evp_official_aligned.model import JointEVPModel
from cod_iml_evp_training_v2.strategy import build_optimizer
from .model import ReferenceEVPModel
from .train import make_dataset, evaluate, seed_everything, state_sha256


def has_gradient(module):
    return any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())


def check_parent_initial_tensors(model, parent):
    current = model.state_dict()
    for name, value in parent.state_dict().items():
        match = re.fullmatch(r"encoder\.task_tunes\.(cod|iml)\.(\d+)\.(\d+)\.(.+)", name)
        target = name
        if match:
            task, stage, block, suffix = match.groups()
            stage, block = int(stage), int(block)
            count = len(model.encoder.shared_tunes[stage])
            target = (f"encoder.shared_tunes.{stage}.{block}.{suffix}" if block < count else
                      f"encoder.private_tunes.{task}.{stage}.{block - count}.{suffix}")
        if target not in current or not torch.equal(value, current[target]):
            raise AssertionError("Parent initialization differs: " + name)
    return len(parent.state_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(2)
    seed_everything(config["seed"])
    model = ReferenceEVPModel(**config["model"])
    loaded = model.load_spatial_pretrained(config["spatial_pretrained"])
    frozen_hash = state_sha256(model.immutable_backbone_state())
    seed_everything(config["seed"])
    baseline = JointEVPModel(**{k: v for k, v in config["model"].items()
                               if k in inspect.signature(JointEVPModel.__init__).parameters})
    baseline.load_spatial_pretrained(config["spatial_pretrained"])
    common = baseline.state_dict()
    current = model.state_dict()
    check_parent_initial_tensors(model, baseline)
    report = {"scope": "Real-data runtime/gradient preflight, not an accuracy experiment",
              "loaded": loaded, "common_initial_tensors_equal": len(common),
              "frozen_backbone_sha256": frozen_hash, "arms": {},
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    args.output.mkdir(exist_ok=True, parents=True)
    batches = {}
    for task in ["cod", "iml"]:
        dataset = make_dataset(config, task, "train")
        batch = next(iter(DataLoader(Subset(dataset, range(4)), batch_size=4, num_workers=0)))
        batches[task] = {k: batch[k].cuda() for k in ["image", "mask"]}
    baseline.cuda().eval()
    model.cuda().eval()
    with torch.no_grad():
        errors = {}
        for task in ["cod", "iml"]:
            x = batches[task]["image"][:1]
            a, b = model(x, task=task)[task], baseline(x, task=task)[task]
            errors[task] = float((a - b).abs().max())
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    report["initial_logit_max_error"] = errors
    del baseline, common, current
    gc.collect()
    torch.cuda.empty_cache()
    optimizer, groups = build_optimizer(model, config)
    report["optimizer_groups"] = groups
    for arm, tasks in [("cod_only", ["cod"]), ("iml_only", ["iml"]), ("joint", ["cod", "iml"])]:
        times = []
        torch.cuda.reset_peak_memory_stats()
        roles = model.parameter_groups()
        foreign = "iml" if tasks == ["cod"] else "cod" if tasks == ["iml"] else None
        foreign_before = state_sha256({str(i): p for i, p in enumerate(roles[foreign + "_private"])}) if foreign else None
        for _ in range(2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            logs = joint_train_step(model, optimizer,
                **{t + "_batch": batches[t] for t in tasks}, amp=True, max_grad_norm=5.,
                iml_dice_weight=float(config.get("iml_dice_weight",0.)), clip_mode=config.get("clip_mode","global"))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        for module in [model.encoder.shared_tunes, model.encoder.prompt_generator]:
            if any(p.requires_grad for p in module.parameters()):
                assert has_gradient(module), type(module).__name__
        for task in tasks:
            assert has_gradient(model.encoder.private_tunes[task])
        if foreign:
            assert all(p.grad is None for p in roles[foreign + "_private"])
            assert foreign_before == state_sha256({str(i): p for i, p in enumerate(roles[foreign + "_private"])})
        assert all(p.grad is None for p in model.encoder.backbone.parameters() if not p.requires_grad)
        entry = {"seconds": times, "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                 "losses": {k: float(v) for k, v in logs.items()},
                 "foreign_private_unchanged": foreign is not None}
        report["arms"][arm] = entry
        (args.output / "audit.json").write_text(json.dumps(report, indent=2))
    assert state_sha256(model.immutable_backbone_state()) == frozen_hash
    report["frozen_backbone_unchanged"] = True
    # Exercise the inherited full evaluator on real held-out images. No new
    # threshold is selected and these small-subset numbers are not run results.
    report["evaluation_smoke"] = {}
    for task in ["cod", "iml"]:
        validation = Subset(make_dataset(config, task, "val"), range(4))
        result = evaluate(model, DataLoader(validation, batch_size=1, num_workers=0),
                          task, torch.device("cuda"), amp=False,
                          iml_dice_weight=float(config.get("iml_dice_weight",0.)))
        report["evaluation_smoke"][task] = {k: result[k] for k in
            (["smeasure", "weighted_fmeasure"] if task == "cod" else ["tampered_f1", "oracle_f1_macro"])}
    report["state"] = "passed"
    (args.output / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
