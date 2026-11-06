"""Shared Q/KV learning and lossless sparse-checkpoint reconstruction."""
import pytest
import torch

from cod_iml_evp_reference.model import ReferenceEVPModel
from cod_iml_evp_full_image.losses import task_loss
from cod_iml_evp_full_image.lora import SharedLowRankLinear
from cod_iml_evp_training_v2.strategy import build_optimizer

SMALL = dict(channels=(8, 16, 32, 64), depths=(1, 1, 1, 1), heads=(1, 2, 4, 8),
             sr_ratios=(8, 4, 2, 1), decoder_width=16, drop_path_rate=0., checkpoint_blocks=False)


def generic_weights(tmp_path):
    model = ReferenceEVPModel(**SMALL)
    path = tmp_path / "generic.pth"
    torch.save(model.encoder.backbone.state_dict(), path)
    return path


def lora_parameters(model):
    return {name: p for name, p in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}


def test_attention_update_has_exact_parent_initial_logits_and_optimizer_coverage():
    torch.manual_seed(17)
    original = ReferenceEVPModel(**SMALL).eval()
    torch.manual_seed(17)
    adapted = ReferenceEVPModel(**SMALL, attention_lora_rank=4).eval()
    assert len(adapted.attention_lora_modules) == 8
    image = torch.rand(1, 3, 64, 80)
    for task in ("cod", "iml"):
        with torch.no_grad():
            assert torch.equal(original(image, task=task)[task], adapted(image, task=task)[task])
    base = original.encoder.backbone.state_dict()
    current = adapted.encoder.backbone.state_dict()
    for key in base:
        assert torch.equal(base[key], current[key])
    optimizer, _ = build_optimizer(adapted, {})
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in adapted.parameters() if p.requires_grad}
    shared = {id(p) for p in adapted.parameter_groups()["shared"]}
    assert {id(p) for p in lora_parameters(adapted).values()} <= shared


@pytest.mark.parametrize("task", ["cod", "iml"])
def test_both_tasks_update_the_same_attention_weights_and_not_foreign_private(tmp_path, task):
    torch.manual_seed(2)
    weights = generic_weights(tmp_path)
    model = ReferenceEVPModel(**SMALL, attention_lora_rank=4).eval()
    loaded = model.load_spatial_pretrained(weights)
    assert loaded["new_low_rank_keys"] == 16
    immutable = {k: v.clone() for k, v in model.immutable_backbone_state().items()}
    roles = model.parameter_groups()
    foreign = "iml" if task == "cod" else "cod"
    before = [p.detach().clone() for p in roles[foreign + "_private"]]
    optimizer, _ = build_optimizer(model, {"shared_lr": 2e-4, "weight_decay": 0.})
    image = torch.rand(2, 3, 64, 80)
    target = torch.randint(0, 2, (2, 1, 64, 80)).float()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(image, task=task)[task]
    task_loss(task, prediction, target).backward()
    low_rank = lora_parameters(model)
    for name, p in low_rank.items():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        if name.endswith(".lora_A"):
            assert p.grad.count_nonzero() == 0  # Correct on the first step: B=0.
        else:
            assert p.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in roles[foreign + "_private"])
    optimizer.step()
    assert any(p.abs().sum() > 0 for name, p in low_rank.items() if name.endswith(".lora_B"))
    for p, value in zip(roles[foreign + "_private"], before):
        assert torch.equal(p, value)
    for key, value in model.immutable_backbone_state().items():
        assert torch.equal(value, immutable[key])
    optimizer.zero_grad(set_to_none=True)
    task_loss(task, model(image, task=task)[task], target).backward()
    assert any(p.grad.abs().sum() > 0 for name, p in low_rank.items() if name.endswith(".lora_A"))


def test_sparse_checkpoint_retains_learned_backbone_updates_and_restores_exactly(tmp_path):
    weights = generic_weights(tmp_path)
    model = ReferenceEVPModel(**SMALL, attention_lora_rank=4, frequency_fusion="task_gate").eval()
    model.load_spatial_pretrained(weights)
    with torch.no_grad():
        for name, p in lora_parameters(model).items():
            if name.endswith(".lora_B"):
                p.normal_(std=.01)
    state = model.training_state()
    assert any(key.startswith("encoder.backbone.") and key.endswith(".lora_B") for key in state)
    assert not (set(state) & set(model.immutable_backbone_state()))
    checkpoint = tmp_path / "learned.pt"
    torch.save({"model": state, "generic_sha256": model._pretrained_sha256, "config": model.config}, checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    restored = ReferenceEVPModel(**saved["config"]).eval()
    restored.load_spatial_pretrained(weights)
    restored.restore_training_state(saved["model"], saved["generic_sha256"])
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    image = torch.rand(1, 3, 64, 80)
    with torch.no_grad():
        for task in ("cod", "iml"):
            assert torch.equal(model(image, task=task)[task], restored(image, task=task)[task])
    broken = dict(state)
    broken.pop(next(key for key in broken if key.endswith(".lora_B")))
    with pytest.raises(ValueError, match="learned tensors"):
        restored.restore_training_state(broken, saved["generic_sha256"])
    with pytest.raises(ValueError, match="hash differs"):
        restored.restore_training_state(state, "wrong initialization")


@pytest.mark.parametrize("defect", ["missing_generic", "supplied_lora", "unexpected_key"])
def test_generic_loading_rejects_every_undeclared_difference(tmp_path, defect):
    weights = generic_weights(tmp_path)
    source = torch.load(weights, weights_only=False)
    model = ReferenceEVPModel(**SMALL, attention_lora_rank=4)
    if defect == "missing_generic":
        source.pop(next(iter(source)))
    elif defect == "supplied_lora":
        source["block1.0.attn.q.lora_A"] = torch.zeros(4, 8)
    else:
        source["another_backbone.weight"] = torch.zeros(1)
    torch.save(source, weights)
    with pytest.raises(ValueError, match="checkpoint keys differ"):
        model.load_spatial_pretrained(weights)


def test_generic_wrappers_and_classifier_head_are_handled_explicitly(tmp_path):
    weights = generic_weights(tmp_path)
    source = {"module.backbone." + key: value for key, value in torch.load(weights, weights_only=False).items()}
    source["module.backbone.head.weight"] = torch.zeros(1000, 64)
    torch.save({"state_dict": source}, weights)
    model = ReferenceEVPModel(**SMALL, attention_lora_rank=4)
    report = model.load_spatial_pretrained(weights)
    assert report["ignored_keys"] == ["head.weight"]
    assert report["new_low_rank_keys"] == 16


def test_attention_rank_options_reject_invalid_or_unfrozen_base():
    for rank in (True, -1, 2.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            ReferenceEVPModel(**SMALL, attention_lora_rank=rank)
    with pytest.raises(ValueError, match="frozen generic"):
        ReferenceEVPModel(**SMALL, attention_lora_rank=4, freeze_backbone=False)
    with pytest.raises(ValueError, match="alpha"):
        ReferenceEVPModel(**SMALL, attention_lora_rank=4, attention_lora_alpha=float("nan"))
    for module in ReferenceEVPModel(**SMALL, attention_lora_rank=4).modules():
        if isinstance(module, SharedLowRankLinear):
            assert module.lora_A.abs().sum() > 0 and module.lora_B.count_nonzero() == 0


@pytest.mark.parametrize("sharing", ["all_private", "stage_last_private"])
def test_legacy_preflight_maps_original_prompt_names_correctly(sharing):
    from cod_iml_evp_reference.audit import check_parent_initial_tensors
    from cod_iml_evp_official_aligned.model import JointEVPModel
    torch.manual_seed(7)
    parent = JointEVPModel(**SMALL)
    torch.manual_seed(7)
    model = ReferenceEVPModel(**SMALL, sharing=sharing, attention_lora_rank=4)
    assert check_parent_initial_tensors(model, parent) == len(parent.state_dict())


def test_sparse_restore_shape_error_cannot_partially_modify_the_model(tmp_path):
    weights = generic_weights(tmp_path)
    model = ReferenceEVPModel(**SMALL, attention_lora_rank=4)
    model.load_spatial_pretrained(weights)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    broken = {k: v.clone() for k, v in model.training_state().items()}
    broken[next(k for k in broken if k.endswith(".lora_B"))].add_(1.)
    key = next(k for k in broken if k.endswith(".lora_A"))
    broken[key] = broken[key][:, :-1]
    with pytest.raises(ValueError, match="shape/dtype"):
        model.restore_training_state(broken, model._pretrained_sha256)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
