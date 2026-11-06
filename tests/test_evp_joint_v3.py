"""Independent optimizer equivalence, readout routing and checkpoint integrity."""
import copy

import pytest
import torch

from cod_iml_evp_reference.model import ReferenceEVPModel
from cod_iml_evp_joint_v3.model import JointStrategyModel
from cod_iml_evp_joint_v3.optimizer import TaskAdamW, build_optimizer
from cod_iml_evp_joint_v3.engine import joint_train_step
from cod_iml_evp_full_image.losses import task_loss

SMALL = dict(channels=(8, 16, 32, 64), depths=(2, 2, 2, 2), heads=(1, 2, 4, 8),
             sr_ratios=(8, 4, 2, 1), decoder_width=16, drop_path_rate=0., checkpoint_blocks=False,
             attention_lora_rank=4)


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_tree(x, y)
    else:
        assert a == b


def group(p, role="shared", lr=.003, wd=0.):
    return dict(params=[p], role=role, name=role, lr=lr, weight_decay=wd)


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_single_task_matches_torch_adamw_including_private_roles_and_decay(aggregation):
    x = torch.nn.Parameter(torch.tensor([1., -2., 3.], dtype=torch.float64))
    y = torch.nn.Parameter(x.detach().clone())
    p = torch.nn.Parameter(torch.tensor([2., 4., -3.], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    opt = TaskAdamW([group(x, wd=.05), group(p, "cod_private", lr=.007, wd=.03)], aggregation=aggregation)
    baseline = torch.optim.AdamW([dict(params=[y], lr=.003, weight_decay=.05),
                                 dict(params=[q], lr=.007, weight_decay=.03)], foreach=False)
    for i in range(7):
        gx = torch.tensor([.3 + i, -1. / (i + 1), (-1.) ** i], dtype=torch.float64)
        gp = gx * .2
        x.grad, y.grad, p.grad, q.grad = gx.clone(), gx.clone(), gp.clone(), gp.clone()
        opt.step(task_gradients={"cod": [gx]})
        baseline.step()
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_two_task_adaptive_proposals_match_independent_adam_references(aggregation):
    p = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.float64))
    reference_p = p.detach().clone()
    replicas = [torch.nn.Parameter(p.detach().clone()) for _ in range(2)]
    opts = [torch.optim.AdamW([replica], lr=.003, weight_decay=0., foreach=False) for replica in replicas]
    opt = TaskAdamW([group(p, wd=.1)], aggregation=aggregation)
    for i in range(8):
        gradients = [torch.tensor([1. + i, (-1.) ** i], dtype=torch.float64),
                     torch.tensor([-.02 * (i + 1), .3], dtype=torch.float64)]
        proposals = []
        for replica, reference, grad in zip(replicas, opts, gradients):
            old = replica.detach().clone()
            replica.grad = grad
            reference.step()
            proposals.append(replica.detach() - old)
        reference_p = reference_p * (1 - .003 * .1) + sum(proposals) / (2 if aggregation == "mean" else 1)
        opt.step(task_gradients={"iml": [gradients[1]], "cod": [gradients[0]]})
        torch.testing.assert_close(p, reference_p, rtol=1e-12, atol=1e-12)
    assert opt.state[p]["task_moments"]["cod"]["step"] == 8
    assert opt.state[p]["task_moments"]["iml"]["step"] == 8


def test_optimizer_restores_every_moment_counter_and_rejects_wrong_schema():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    opt = TaskAdamW([group(p)])
    opt.step(task_gradients={"cod": [torch.tensor([1., 2.])], "iml": [torch.tensor([-.3, .5])]})
    q = torch.nn.Parameter(p.detach().clone())
    restored = TaskAdamW([group(q)])
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for t in range(3):
        gradients = {"cod": [torch.tensor([.2, -.1])], "iml": [torch.tensor([.3, .4])]}
        opt.step(task_gradients=gradients)
        restored.step(task_gradients=gradients)
    assert torch.equal(p, q)
    same_tree(opt.state_dict(), restored.state_dict())
    with pytest.raises(ValueError, match="schema"):
        TaskAdamW([group(q)], aggregation="mean").load_state_dict(opt.state_dict())


def test_missing_task_is_not_decayed_and_nonfinite_step_is_transactionally_rejected():
    p, private = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
    opt = TaskAdamW([group(p, wd=.3), group(private, "iml_private", wd=.3)])
    original = private.detach().clone()
    opt.step(task_gradients={"cod": [torch.ones(2)]})
    assert torch.equal(private, original) and private not in opt.state
    before, states = p.detach().clone(), copy.deepcopy(opt.state_dict())
    with pytest.raises(FloatingPointError):
        opt.step(task_gradients={"cod": [torch.ones(2)], "iml": [torch.tensor([float("nan"), 0.])]})
    assert torch.equal(p, before)
    same_tree(states, opt.state_dict())


@pytest.mark.parametrize("exits", [False, True])
def test_initial_logits_and_all_common_tensors_equal_current_qkv(exits):
    torch.manual_seed(42)
    parent = ReferenceEVPModel(**SMALL).eval()
    torch.manual_seed(42)
    model = JointStrategyModel(**SMALL, private_stage_exits=exits).eval()
    for key, value in parent.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    x = torch.rand(2, 3, 64, 80)
    with torch.no_grad():
        for task in ("cod", "iml"):
            a = parent(x, task=task, processed_sizes=[(50, 70), (64, 80)])[task]
            b = model(x, task=task, processed_sizes=[(50, 70), (64, 80)])[task]
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("task", ["cod", "iml"])
def test_private_exits_never_feed_deeper_shared_stages_and_foreign_exits_are_isolated(task):
    model = JointStrategyModel(**SMALL, private_stage_exits=True).eval()
    image = torch.rand(1, 3, 64, 80)
    before = {key: value.clone() for key, value in model.encoder.backbone.state_dict().items()}
    features = model._features(image, task)
    # Only the deepest export is supervised: earlier private exports cannot
    # influence the shared stream which supplies this deepest representation.
    features[-1].square().mean().backward()
    for readout in model.encoder.exit_readouts[task][:-1]:
        assert all(p.grad is None for p in readout.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.exit_readouts[task][-1].parameters())
    model.zero_grad(set_to_none=True)
    prediction = model(image, task=task)[task]
    task_loss(task, prediction, torch.randint(0, 2, prediction.shape).float()).backward()
    for readout in model.encoder.exit_readouts[task]:
        assert readout.attn_proj.lora_B.grad.abs().sum() > 0
        assert readout.fc1.lora_B.grad.abs().sum() > 0
        assert readout.fc2.lora_B.grad.abs().sum() > 0
    foreign = "iml" if task == "cod" else "cod"
    assert all(p.grad is None for p in model.encoder.exit_readouts[foreign].parameters())
    for key, value in model.encoder.backbone.state_dict().items():
        assert torch.equal(value, before[key]), key  # No merged matrix mutation.


def test_private_exit_checkpoint_recomputation_matches_noncheckpointed_gradients():
    torch.manual_seed(9)
    a = JointStrategyModel(**SMALL, private_stage_exits=True).train()
    b = JointStrategyModel(**dict(SMALL, checkpoint_blocks=True), private_stage_exits=True).train()
    with torch.no_grad():
        for key, p in a.named_parameters():
            if key.endswith(".lora_B"):
                p.normal_(std=.01)
    b.load_state_dict(a.state_dict())
    x = torch.rand(2, 3, 64, 80)
    gt = torch.randint(0, 2, (2, 1, 64, 80)).float()
    torch.manual_seed(72)
    out_a = a(x, task="cod")["cod"]
    task_loss("cod", out_a, gt).backward()
    torch.manual_seed(72)
    out_b = b(x, task="cod")["cod"]
    task_loss("cod", out_b, gt).backward()
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
    for (key, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert (p.grad is None) == (q.grad is None), key
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=key)


@pytest.mark.parametrize("mode", ["adamw", "task_adamw_sum", "task_adamw_mean"])
def test_joint_engine_correct_roles_finite_gradients_and_sparse_checkpoint(tmp_path, mode):
    torch.manual_seed(31)
    model = JointStrategyModel(**SMALL, private_stage_exits=True)
    generic = tmp_path / "generic.pth"
    base = ReferenceEVPModel(**dict(SMALL, attention_lora_rank=0))
    torch.save(base.encoder.backbone.state_dict(), generic)
    model.load_spatial_pretrained(generic)
    immutable = {key: v.clone() for key, v in model.immutable_backbone_state().items()}
    opt, _ = build_optimizer(model, dict(shared_optimizer=mode))
    params = [p for group in opt.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    shared_ids = {id(p) for p in model.parameter_groups()["shared"]}
    assert not shared_ids & {id(p) for p in model.encoder.exit_readouts.parameters()}
    batch = dict(image=torch.rand(2, 3, 64, 80), mask=torch.randint(0, 2, (2, 1, 64, 80)).float())
    logs = joint_train_step(model, opt, cod_batch=batch, iml_batch=batch)
    assert all(torch.isfinite(value) for value in logs.values())
    for key, value in model.immutable_backbone_state().items():
        assert torch.equal(value, immutable[key])
    restored = JointStrategyModel(**model.config)
    restored.load_spatial_pretrained(generic)
    restored.restore_training_state(model.training_state(), model._pretrained_sha256)
    same_tree(model.state_dict(), restored.state_dict())


def test_nonfinite_second_task_makes_no_optimizer_update():
    model = JointStrategyModel(**SMALL)
    optimizer, _ = build_optimizer(model, dict(shared_optimizer="task_adamw_sum"))
    before = {key: p.clone() for key, p in model.named_parameters()}
    batch = dict(image=torch.rand(2, 3, 64, 80), mask=torch.zeros(2, 1, 64, 80))
    invalid = dict(batch, mask=torch.full((2, 1, 64, 80), float("nan")))
    with pytest.raises(FloatingPointError):
        joint_train_step(model, optimizer, cod_batch=batch, iml_batch=invalid)
    assert not optimizer.state
    for key, p in model.named_parameters():
        assert torch.equal(p, before[key])


def test_bf16_private_update_is_not_merged_into_rounded_generic_weights():
    from cod_iml_evp_joint_v3.model import LowRankDelta
    from torch.nn import functional as F
    layer = torch.nn.Linear(2, 1, bias=False)
    layer.requires_grad_(False)
    with torch.no_grad():layer.weight.fill_(.5)
    delta = LowRankDelta(layer, 1, 1.)
    with torch.no_grad():
        delta.lora_A.copy_(torch.tensor([[1., 0.]]))
        delta.lora_B.fill_(1e-4)
    x = torch.tensor([[1., -1.]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        separate = layer(x) + delta(x)
        merged = F.linear(x, layer.weight + delta())
    assert separate.abs().item() > 0
    assert merged.item() == 0


def test_zero_private_exits_preserve_training_logits_and_rng_with_drop_path():
    config = dict(SMALL, drop_path_rate=.15)
    torch.manual_seed(42)
    parent = ReferenceEVPModel(**config).train()
    torch.manual_seed(42)
    model = JointStrategyModel(**config, private_stage_exits=True).train()
    image = torch.rand(2, 3, 64, 80)
    torch.manual_seed(712)
    a = parent(image, task="cod")["cod"]
    original_rng = torch.get_rng_state().clone()
    torch.manual_seed(712)
    b = model(image, task="cod")["cod"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(original_rng, torch.get_rng_state())


@pytest.mark.parametrize("mode,exits", [("task_adamw_sum", False), ("adamw", True), ("task_adamw_sum", True)])
def test_complete_training_and_epoch_recovery_reproduce_model_optimizer_scheduler(tmp_path, monkeypatch, mode, exits):
    import json
    import sys
    import numpy as np
    from PIL import Image
    from cod_iml_evp_joint_v3 import train as trainer
    data = {}
    for task in ("cod", "iml"):
        directory = tmp_path / task
        directory.mkdir()
        rows = []
        for i, shape in enumerate(((45, 60), (60, 50))):
            rng = np.random.default_rng(i)
            image = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
            mask = np.zeros(shape, dtype=np.uint8);mask[10:30, 10:30] = 255
            Image.fromarray(image).save(directory / f"{i}.png")
            Image.fromarray(mask).save(directory / f"{i}_gt.png")
            rows.append(dict(id=str(i), image=f"{i}.png", mask=f"{i}_gt.png", task=task, family="splicing"))
        path = directory / "samples.jsonl";path.write_text("\n".join(json.dumps(r) for r in rows))
        data[task] = dict(train=str(path), val=str(path))
    generic = tmp_path / "generic.pth"
    torch.save(ReferenceEVPModel(**dict(SMALL, attention_lora_rank=0)).encoder.backbone.state_dict(), generic)
    base = dict(arm="joint", model=dict(SMALL, private_stage_exits=exits), model_variant="evp_joint_v3",
        shared_optimizer=mode, data=data, spatial_pretrained=str(generic), epochs=2, steps_per_epoch=1,
        seed=42, device="cpu", batch_size=2, workers=0, amp=False, eval_amp=False, evaluate_test=False,
        cod_metrics=False, warmup_epochs=0., max_grad_norm=5., archive_epoch_checkpoints=True,
        input_geometry={t:dict(max_side=64,resize_mode="native") for t in data},
        selection_metric={"cod":"f1","iml":"tampered_f1"}, frozen_package_sources={},
        casia_evaluation_runner_sha256=trainer.file_sha256(sys.modules[trainer.add_tampered_metrics.__module__].__file__))
    def config(name):
        path = tmp_path / (name + ".json");path.write_text(json.dumps(dict(base, output_dir=str(tmp_path / name))))
        return trainer.resolve_config(path)
    trainer.run_experiment(config("straight"))
    original = trainer.atomic_checkpoint
    class Interrupted(RuntimeError):
        pass
    def interrupt(path, payload):
        original(path, payload)
        if payload["completed_epoch"] == 1:
            raise Interrupted("intentional checkpoint-boundary interruption")
    with monkeypatch.context() as patch:
        patch.setattr(trainer,"atomic_checkpoint",interrupt)
        with pytest.raises(Interrupted):trainer.run_experiment(config("resumed"))
    trainer.run_experiment(config("resumed"),resume=tmp_path / "resumed/last.pt")
    a=torch.load(tmp_path / "straight/last.pt",weights_only=False)
    b=torch.load(tmp_path / "resumed/last.pt",weights_only=False)
    assert a["model_sha256"]==b["model_sha256"]
    same_tree(a["optimizer"],b["optimizer"])
    same_tree(a["scheduler"],b["scheduler"])
    assert (tmp_path / "resumed/epoch_002.pt").exists()


def test_held_out_tests_are_only_opened_after_training_and_reuse_identical_selected_checkpoint(tmp_path, monkeypatch):
    import json
    import sys
    import numpy as np
    from PIL import Image
    from cod_iml_evp_joint_v3 import train as trainer
    from cod_iml_evp_full_image.data import FullImageDataset
    data = {}
    for task in ("cod", "iml"):
        folder = tmp_path / task
        folder.mkdir()
        # These are synthetic software fixtures, not a scientific data split.
        image = np.random.default_rng(32).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        mask = np.zeros((48, 64), dtype=np.uint8);mask[10:35, 12:40] = 255
        Image.fromarray(image).save(folder / "image.png")
        Image.fromarray(mask).save(folder / "mask.png")
        specs = {}
        for split in ("train", "val", "test"):
            row = dict(id=split, image="image.png", mask="mask.png", task=task, family="splicing")
            path = folder / (split + ".jsonl");path.write_text(json.dumps(row))
            specs[split] = str(path)
        data[task] = specs
    generic = tmp_path / "generic.pth"
    torch.save(ReferenceEVPModel(**dict(SMALL, attention_lora_rank=0)).encoder.backbone.state_dict(), generic)
    config = dict(arm="joint", model=dict(SMALL, private_stage_exits=True), model_variant="evp_joint_v3",
        shared_optimizer="task_adamw_sum", data=data, spatial_pretrained=str(generic),
        output_dir=str(tmp_path / "run"), epochs=1, steps_per_epoch=1, batch_size=2,
        seed=42, device="cpu", workers=0, amp=False, eval_amp=False, evaluate_test=True,
        cod_metrics=False, warmup_epochs=0., max_grad_norm=5.,
        input_geometry={t:dict(max_side=64,resize_mode="native") for t in data},
        selection_metric={"cod":"f1","iml":"tampered_f1"}, frozen_package_sources={},
        casia_evaluation_runner_sha256=trainer.file_sha256(sys.modules[trainer.add_tampered_metrics.__module__].__file__))
    accesses = []
    original = FullImageDataset.__getitem__
    def checked_getitem(dataset, draw):
        if dataset.manifest.name == "test.jsonl":
            status = json.loads((tmp_path / "run/status.json").read_text())
            assert status["state"] == "final_testing" and status["completed_epochs"] == 1
            accesses.append(dataset.task)
        return original(dataset, draw)
    monkeypatch.setattr(FullImageDataset, "__getitem__", checked_getitem)
    path = tmp_path / "config.json";path.write_text(json.dumps(config))
    trainer.run_experiment(trainer.resolve_config(path))
    result = json.loads((tmp_path / "run/test_results.json").read_text())
    assert sorted(accesses) == ["cod", "iml"]
    for task in ("cod", "iml"):
        assert result[task]["final"]["datasets"]["test"]["images"] == 1
        assert result[task]["best_val"]["datasets"]["test"]["reused_identical_checkpoint_evaluation"]
