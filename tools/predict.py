"""Restore a sparse research checkpoint and predict one complete image."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from cod_iml_evp_joint_v3.model import JointStrategyModel
from cod_iml_evp_full_image.data import pad_collate
from cod_iml_evp_full_image.geometry import restore_logits


def predict(checkpoint, pretrained, image_path, task, *, device="cpu", amp=False):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("phase") != "epoch_complete":
        raise ValueError("expected a complete-epoch training checkpoint")
    if task not in saved["config"].get("selection_metric", {}):
        raise ValueError("task is absent from checkpoint configuration")
    arm = saved["config"]["arm"]
    if arm != "joint" and arm != task + "_only":
        raise ValueError("requested task was not supervised in this single-task checkpoint")
    model = JointStrategyModel(**saved["model_config"])
    model.load_spatial_pretrained(pretrained)
    model.restore_training_state(saved["model"], saved["immutable_pretrained_sha256"])
    device = torch.device(device)
    if amp and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("--amp requires a BF16-capable CUDA device")
    model.to(device).eval()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    original_h, original_w = image.height, image.width
    geometry = saved["config"]["input_geometry"][task]
    scale = geometry.get("max_side", max(image.size)) / max(image.size)
    if geometry["resize_mode"] == "native":
        scale = min(1., scale)
    h, w = max(1, round(original_h * scale)), max(1, round(original_w * scale))
    image = image.resize((w, h), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2,0,1) / 255.)
    batch = pad_collate([dict(image=tensor,mask=torch.zeros(1,h,w),processed_size=(h,w))])
    with torch.inference_mode():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(batch["image"].to(device),task=task,processed_sizes=batch["processed_size"])[task]
        # Only the shape is used; no ground-truth mask is needed for prediction.
        logits = restore_logits(logits,batch["processed_size"],[torch.empty(original_h, original_w)])[0]
        return logits.sigmoid()[0,0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--task", choices=("cod","iml"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="PNG probability map, without min-max normalization")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true", help="use BF16 on a supported CUDA device")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be a PNG file")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    probability = predict(args.checkpoint,args.pretrained,args.image,args.task,device=args.device,amp=args.amp)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(probability * 255).astype(np.uint8)).save(args.output)


if __name__ == "__main__":
    main()
