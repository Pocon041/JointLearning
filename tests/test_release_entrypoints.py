"""Portable configs and native-size inference must match the frozen trainer."""
import json

import numpy as np
from PIL import Image
import pytest
import torch

from cod_iml_evp_reference.model import ReferenceEVPModel
from cod_iml_evp_joint_v3 import train
from cod_iml_evp_full_image.data import pad_collate
from cod_iml_evp_full_image.geometry import restore_logits
from jointlearning import build_model
from tools.predict import predict
from tools.prepare_configs import make_configs


@pytest.mark.parametrize("stage,epochs,steps,warmup", [
    ("quick",6,352,105), ("expanded",6,1153,343), ("full",30,1153,343)])
def test_generated_configs_keep_matched_protocol_and_validate_sources(tmp_path, stage, epochs, steps, warmup):
    configs = make_configs(stage=stage,pretrained=tmp_path/"generic.pth",
                           manifest_root=tmp_path/"manifests",run_root=tmp_path/"runs")
    assert set(configs) == {"both_cod_only","both_iml_only","both_joint"}
    for name, config in configs.items():
        path = tmp_path / (name+".json")
        path.write_text(json.dumps(config))
        config = train.resolve_config(path)
        train.validate_reference_source(config)
        assert config["epochs"] == epochs and config["steps_per_epoch"] == steps
        assert int(config["warmup_epochs"]*steps) == warmup
        assert config["evaluate_test"] == (stage == "full")
        assert config["model"]["private_stage_exits"]
        assert config["shared_optimizer"] == "task_adamw_sum"
        assert config["data"]["iml"]["train"].endswith(
            "train_quick.jsonl" if stage == "quick" else "train_full.jsonl")
    normalized = [{k:v for k,v in config.items() if k not in ("arm","output_dir")}
                  for config in configs.values()]
    assert normalized[0] == normalized[1] == normalized[2]


@pytest.mark.parametrize("task", ["cod","iml"])
def test_prediction_restores_checkpoint_geometry_and_exact_probabilities(tmp_path,task):
    small = dict(channels=(8,16,32,64),depths=(1,1,1,1),heads=(1,2,4,8),
        sr_ratios=(8,4,2,1),decoder_width=16,drop_path_rate=0.,checkpoint_blocks=False,
        attention_lora_rank=4)
    generic = tmp_path / "generic.pth"
    torch.save(ReferenceEVPModel(**dict(small,attention_lora_rank=0)).encoder.backbone.state_dict(),generic)
    model = build_model(**small).eval()
    model.load_spatial_pretrained(generic)
    saved = dict(phase="epoch_complete",model_config=model.config,model=model.training_state(),
        immutable_pretrained_sha256=model._pretrained_sha256,
        config=dict(arm="joint",selection_metric={"cod":"smeasure","iml":"tampered_f1"},
        input_geometry={"cod":dict(max_side=48,resize_mode="fit"),
                        "iml":dict(max_side=48,resize_mode="native")}))
    checkpoint=tmp_path/"checkpoint.pt";torch.save(saved,checkpoint)
    image = np.random.default_rng(7).integers(0,256,(37,59,3),dtype=np.uint8)
    image_path=tmp_path/"image.png";Image.fromarray(image).save(image_path)
    actual=predict(checkpoint,generic,image_path,task)
    h,w=round(37*48/59),48
    resized=Image.fromarray(image).resize((w,h),Image.Resampling.BILINEAR)
    tensor=torch.from_numpy(np.array(resized,dtype=np.float32).transpose(2,0,1)/255.)
    batch=pad_collate([dict(image=tensor,mask=torch.zeros(1,h,w),processed_size=(h,w))])
    with torch.inference_mode():
        logits=model(batch["image"],task=task,processed_sizes=batch["processed_size"])[task]
        expected=restore_logits(logits,batch["processed_size"],[torch.empty(37,59)])[0].sigmoid()[0,0].numpy()
    np.testing.assert_array_equal(actual,expected)
    assert actual.shape == (37,59)
    assert np.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1
    saved["config"]["arm"]="iml_only" if task == "cod" else "cod_only"
    torch.save(saved,checkpoint)
    with pytest.raises(ValueError,match="not supervised"):
        predict(checkpoint,generic,image_path,task)


def test_shipped_full_configs_are_resolvable_and_source_checked():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for path in (root/"configs/full").glob("*.json"):
        config=train.resolve_config(path)
        train.validate_reference_source(config)
        assert config["epochs"] == 30 and config["steps_per_epoch"] == 1153
        assert config["evaluate_test"] is True
