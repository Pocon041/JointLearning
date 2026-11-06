"""Disposable full-resolution CUDA preflight for the shared-Q/KV candidate."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from cod_iml_evp_reference.model import ReferenceEVPModel
from .model import JointStrategyModel
from .optimizer import build_optimizer
from cod_iml_evp_full_image.data import pad_collate
from .engine import joint_train_step
from .train import (build_model, evaluate, make_dataset, move_batch, resolve_config,
                    seed_everything, state_sha256, task_batch_size, validate_reference_source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = resolve_config(args.config)
    validate_reference_source(config)
    if config["model_variant"] != "evp_joint_v3" or config["model"].get("attention_lora_rank", 0) < 1:
        raise ValueError("the candidate CUDA preflight requires nonzero shared attention updates")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    model = build_model(config)
    loaded = model.load_spatial_pretrained(config["spatial_pretrained"])
    frozen_hash = state_sha256(model.immutable_backbone_state())
    initial_hash = state_sha256(model.state_dict())
    seed_everything(config["seed"])
    parent_config = {k: v for k, v in config["model"].items() if k not in ("private_stage_exits", "exit_rank", "exit_alpha")}
    parent_config["attention_lora_rank"] = 0
    parent = ReferenceEVPModel(**parent_config)
    parent.load_spatial_pretrained(config["spatial_pretrained"])
    current = model.state_dict()
    for key, value in parent.state_dict().items():
        assert torch.equal(value, current[key]), key
    extra = set(current) - set(parent.state_dict())
    assert extra and all(key.endswith((".lora_A", ".lora_B")) for key in extra)
    report = dict(scope="Real-data engineering/gradient checks; not an accuracy experiment",
                  state="checking", loaded=loaded, initial_full_model_sha256=initial_hash,
                  unchanged_parent_tensors=len(parent.state_dict()), new_low_rank_tensors=len(extra),
                  low_rank_parameters=sum(p.numel() for k, p in model.named_parameters()
                                          if k.endswith((".lora_A", ".lora_B"))),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), arms={})

    def persist():
        (args.output / "audit.json").write_text(json.dumps(report, indent=2))

    persist()
    batches = {}
    for task in ("cod", "iml"):
        dataset = make_dataset(config, task, "train")
        batch = next(iter(DataLoader(Subset(dataset, range(task_batch_size(config, task))),
            batch_size=task_batch_size(config, task), collate_fn=pad_collate, num_workers=0)))
        batches[task] = move_batch(batch, torch.device("cuda"))
    model.cuda().eval()
    parent.cuda().eval()
    report["initial_logit_max_error"] = {}
    with torch.no_grad():
        for task in ("cod", "iml"):
            batch = batches[task]
            a = model(batch["image"], task=task, processed_sizes=batch["processed_size"])[task]
            b = parent(batch["image"], task=task, processed_sizes=batch["processed_size"])[task]
            report["initial_logit_max_error"][task] = float((a - b).abs().max())
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    del parent, current, a, b
    gc.collect()
    torch.cuda.empty_cache()
    for mode in ("adamw", "task_adamw_sum", "task_adamw_mean"):
        optimizer, groups = build_optimizer(model, dict(config, shared_optimizer=mode))
        report["optimizer_groups"] = groups
        roles = model.parameter_groups()
        shared_ids = {id(p) for p in roles["shared"]}
        low_rank = {key: p for key, p in model.named_parameters() if key.startswith("encoder.backbone.") and key.endswith((".lora_A", ".lora_B"))}
        assert all(id(p) in shared_ids for p in low_rank.values())
        for arm, tasks in (("cod_only", ("cod",)), ("iml_only", ("iml",)), ("joint", ("cod", "iml"))):
            foreign = "iml" if tasks == ("cod",) else "cod" if tasks == ("iml",) else None
            before = state_sha256({str(i): p for i, p in enumerate(roles[foreign + "_private"])}) if foreign else None
            times = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                torch.cuda.synchronize()
                start = time.perf_counter()
                logs = joint_train_step(model, optimizer, **{task + "_batch": batches[task] for task in tasks},
                    amp=bool(config["amp"]), max_grad_norm=config["max_grad_norm"],
                    iml_dice_weight=float(config.get("iml_dice_weight", 0.)), clip_mode=config.get("clip_mode", "global"))
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            stage_norms = {}
            for stage in range(1, 5):
                values = [p.grad.detach().float().norm() for key, p in low_rank.items()
                          if key.startswith(f"encoder.backbone.block{stage}.") and key.endswith(".lora_B") and p.grad is not None]
                assert values and all(torch.isfinite(value) for value in values)
                stage_norms[str(stage)] = float(torch.stack(values).norm())
                assert stage_norms[str(stage)] > 0
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.prompt_generator.parameters())
            assert all(p.grad is None for p in model.encoder.backbone.parameters() if not p.requires_grad)
            if hasattr(model.encoder,"exit_readouts"):
                for task in tasks:
                    for readout in model.encoder.exit_readouts[task]:
                        assert readout.attn_proj.lora_B.grad is not None and readout.attn_proj.lora_B.grad.abs().sum()>0
                        assert readout.fc1.lora_B.grad is not None and readout.fc1.lora_B.grad.abs().sum()>0
                        assert readout.fc2.lora_B.grad is not None and readout.fc2.lora_B.grad.abs().sum()>0
            if foreign:
                assert all(p.grad is None for p in roles[foreign + "_private"])
                assert before == state_sha256({str(i): p for i, p in enumerate(roles[foreign + "_private"])})
            report["arms"][mode + "/" + arm] = dict(seconds=times, peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                shared_attention_B_gradient_norm_by_stage=stage_norms, losses={k: float(v) for k, v in logs.items()},
                foreign_private_unchanged=foreign is not None)
            persist()
    assert state_sha256(model.immutable_backbone_state()) == frozen_hash
    report["immutable_generic_weights_unchanged"] = True
    state = model.training_state()
    assert set(low_rank) <= set(state)
    path = args.output / "disposable_preflight_weights.pt"
    torch.save(dict(model=state, config=model.config, generic_sha256=model._pretrained_sha256), path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    seed_everything(config["seed"])
    restored = JointStrategyModel(**saved["config"])
    restored.load_spatial_pretrained(config["spatial_pretrained"])
    restored.restore_training_state(saved["model"], saved["generic_sha256"])
    for key, value in model.state_dict().items():
        assert torch.equal(value.detach().cpu(), restored.state_dict()[key]), key
    report["checkpoint_all_tensors_equal"] = True
    report["saved_low_rank_tensors"] = len(set(low_rank) & set(saved["model"]))
    del restored, saved, state
    gc.collect()
    report["evaluation_smoke"] = {}
    for task in ("cod", "iml"):
        dataset = Subset(make_dataset(config, task, "val"), range(4))
        result = evaluate(model, DataLoader(dataset, batch_size=1, collate_fn=pad_collate, num_workers=0),
                          task, torch.device("cuda"), amp=bool(config.get("eval_amp", True)))
        report["evaluation_smoke"][task] = {k: result[k] for k in
            (("smeasure", "weighted_fmeasure") if task == "cod" else ("tampered_f1", "oracle_f1_macro"))}
    report["state"] = "passed"
    persist()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
