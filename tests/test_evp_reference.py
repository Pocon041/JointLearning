import pytest
import torch

from cod_iml_evp_official_aligned.model import JointEVPModel
from cod_iml_evp_official_aligned.engine import task_loss
from cod_iml_evp_training_v2.strategy import build_optimizer
from cod_iml_evp_reference.model import ReferenceEVPModel

SMALL = dict(channels=(8, 16, 32, 64), depths=(2, 2, 2, 2), heads=(1, 2, 4, 8),
             sr_ratios=(8, 4, 2, 1), decoder_width=16, drop_path_rate=0., checkpoint_blocks=False)


@pytest.mark.parametrize("sharing", ["all_private", "stage_last_private"])
def test_reference_preserves_initial_evp_computation(sharing):
    torch.manual_seed(42)
    original = JointEVPModel(**SMALL).eval()
    torch.manual_seed(42)
    candidate = ReferenceEVPModel(**SMALL, sharing=sharing).eval()
    x = torch.rand(1, 3, 32, 40)
    with torch.no_grad():
        for task in ["cod", "iml"]:
            torch.testing.assert_close(original(x, task=task)[task], candidate(x, task=task)[task], atol=0, rtol=0)
    old = sum(p.numel() for p in original.parameters())
    new = sum(p.numel() for p in candidate.parameters())
    assert new == old if sharing == "all_private" else new < old


@pytest.mark.parametrize("task", ["cod", "iml"])
def test_shared_stage_prefix_receives_both_losses_and_foreign_private_stays_untouched(task):
    model = ReferenceEVPModel(**SMALL, sharing="stage_last_private")
    pred = model(torch.rand(1, 3, 32, 32), task=task)[task]
    task_loss(task, pred, torch.rand_like(pred)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.shared_tunes.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.private_tunes[task].parameters())
    other = "iml" if task == "cod" else "cod"
    assert all(p.grad is None for p in model.encoder.private_tunes[other].parameters())
    assert all(p.grad is None for p in model.encoder.backbone.parameters())


def test_reference_optimizer_and_frequency_control(tmp_path):
    model = ReferenceEVPModel(**SMALL, sharing="stage_last_private", frequency_enabled=False).eval()
    def forbidden(*args):
        raise AssertionError("No FFT when frequency_enabled=False")
    model.encoder.prompt_generator.init_handcrafted = forbidden
    model(torch.rand(1, 3, 32, 32), task="iml")
    optimizer, _ = build_optimizer(model, {})
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    weights = tmp_path / "generic.pth"
    torch.save(model.encoder.backbone.state_dict(), weights)
    model.load_spatial_pretrained(weights)
    restored = ReferenceEVPModel(**model.config).eval()
    restored.load_spatial_pretrained(weights)
    restored.restore_training_state(model.training_state(), model._pretrained_sha256)
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(x, task="cod")["cod"], restored(x, task="cod")["cod"], atol=0, rtol=0)
