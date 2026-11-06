"""EVP with explicit prompt sharing and optional shared attention adaptation.

stage_last_private borrows the shared-prefix/private-stage-exit principle from
MTLoRA, but operates on EVP prompt MLPs. It is not an MTLoRA reimplementation.
No expert, latent hub, frequency bank, energy threshold or new norm is added.
Q/KV low-rank updates and task-specific HFC consumption are independent options.
"""
import hashlib
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cod_iml_evp_official_aligned.model import JointEVPModel
from cod_iml_evp_full_image.fusion import FrequencyGate
from cod_iml_evp_full_image.geometry import checked_sizes, handcrafted_features_before_padding
from cod_iml_evp_full_image.lora import inject_shared_qkv, load_generic_backbone_with_lora


class ReferencePromptEncoder(nn.Module):
    def __init__(self, base, sharing, frequency_enabled, frequency_fusion):
        super().__init__()
        self.backbone = base.backbone
        self.prompt_generator = base.prompt_generator
        for name in ["channels", "depths", "freeze_backbone", "checkpoint_blocks", "hfc_enabled"]:
            setattr(self, name, getattr(base, name))
        self.sharing = sharing
        self.frequency_enabled = bool(frequency_enabled)
        self.frequency_fusion = frequency_fusion
        self.shared_tunes = nn.ModuleList()
        self.private_tunes = nn.ModuleDict({t: nn.ModuleList() for t in ["cod", "iml"]})
        for s, depth in enumerate(self.depths):
            shared_count = depth - 1 if sharing == "stage_last_private" else 0
            self.shared_tunes.append(nn.ModuleList(list(base.task_tunes["cod"][s])[:shared_count]))
            for task in ["cod", "iml"]:
                self.private_tunes[task].append(nn.ModuleList(list(base.task_tunes[task][s])[shared_count:]))
        if frequency_fusion == "task_gate":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() + 12001)
                self.frequency_gates = nn.ModuleDict({task: nn.ModuleList(
                    FrequencyGate(c // self.prompt_generator.scale_factor) for c in self.channels)
                    for task in ["cod", "iml"]})

    def forward(self, image, task, *, processed_sizes=None):
        hfc = None
        if self.frequency_enabled and self.hfc_enabled:
            hfc = (self.prompt_generator.init_handcrafted(image) if processed_sizes is None else
                   handcrafted_features_before_padding(self.prompt_generator, image, processed_sizes))
        x, features = image, []
        for s in range(1, 5):
            x, h, w = getattr(self.backbone, f"patch_embed{s}")(x)
            content = getattr(self.prompt_generator, f"embedding_generator{s}")(x)
            if hfc is None:
                base = content
            elif self.frequency_fusion == "task_gate":
                base = self.frequency_gates[task][s - 1](content, hfc[s - 1])
            else:
                base = content + hfc[s - 1]
            up = getattr(self.prompt_generator, f"shared_mlp{s}")
            shared_count = len(self.shared_tunes[s - 1])
            for i, block in enumerate(getattr(self.backbone, f"block{s}")):
                tune = self.shared_tunes[s - 1][i] if i < shared_count else self.private_tunes[task][s - 1][i - shared_count]
                prompt = up(tune(base))
                if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                    x = checkpoint(block, x + prompt, h, w, use_reentrant=False, preserve_rng_state=True)
                else:
                    x = block(x + prompt, h, w)
            x = getattr(self.backbone, f"norm{s}")(x).transpose(1, 2).reshape(image.shape[0], -1, h, w)
            features.append(x)
        return tuple(features)


class ReferenceEVPModel(JointEVPModel):
    def __init__(self, *, enhancement="evp", sharing="all_private", frequency_enabled=True,
                 frequency_fusion="sum", attention_lora_rank=0, attention_lora_alpha=None, **base_config):
        if enhancement != "evp" or sharing not in ["all_private", "stage_last_private"]:
            raise ValueError("Reference uses original EVP and an explicit sharing policy")
        if frequency_fusion not in ("sum", "task_gate"):
            raise ValueError("frequency_fusion must be sum or task_gate")
        if isinstance(attention_lora_rank, bool) or not isinstance(attention_lora_rank, int) or attention_lora_rank < 0:
            raise ValueError("attention_lora_rank must be a nonnegative integer")
        alpha = float(attention_lora_rank or 1) if attention_lora_alpha is None else float(attention_lora_alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("attention_lora_alpha must be finite and positive")
        if attention_lora_rank and not base_config.get("freeze_backbone", True):
            raise ValueError("this candidate requires frozen generic backbone weights")
        super().__init__(enhancement="evp", **base_config)
        self.encoder = ReferencePromptEncoder(self.encoder, sharing, frequency_enabled, frequency_fusion)
        self.attention_lora_modules = []
        if attention_lora_rank:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() + 16001)
                self.attention_lora_modules = inject_shared_qkv(self.encoder.backbone, attention_lora_rank, alpha)
        self.config.update(sharing=sharing, frequency_enabled=bool(frequency_enabled), frequency_fusion=frequency_fusion,
                           attention_lora_rank=attention_lora_rank, attention_lora_alpha=alpha)

    def load_spatial_pretrained(self, path, source="auto"):
        if not self.config["attention_lora_rank"]:
            return super().load_spatial_pretrained(path, source=source)
        result = load_generic_backbone_with_lora(self.encoder.backbone, path)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        self._pretrained_sha256 = digest.hexdigest()
        return result

    def _immutable_backbone_keys(self):
        if not self.encoder.freeze_backbone:
            return set()
        # Save ALL buffers and all learned tensors, even when they live under
        # encoder.backbone. A directory prefix does not establish immutability.
        return {"encoder.backbone." + name for name, p in self.encoder.backbone.named_parameters()
                if not p.requires_grad}

    def immutable_backbone_state(self):
        keys = self._immutable_backbone_keys()
        return {key: value for key, value in self.state_dict().items() if key in keys}

    def training_state(self):
        if self.encoder.freeze_backbone and self._pretrained_sha256 is None:
            raise ValueError("Sparse checkpoints require verified generic pretrained weights")
        immutable = self._immutable_backbone_keys()
        return {key: value.detach().cpu() for key, value in self.state_dict().items() if key not in immutable}

    def restore_training_state(self, state, pretrained_sha256):
        if self.encoder.freeze_backbone and self._pretrained_sha256 != pretrained_sha256:
            raise ValueError("Immutable pretrained checkpoint hash differs")
        expected = set(self.state_dict()) - self._immutable_backbone_keys()
        if set(state) != expected:
            raise ValueError("Sparse checkpoint misses learned tensors or contains unexpected tensors")
        current = self.state_dict()
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != current[key].shape or value.dtype != current[key].dtype:
                raise ValueError("Sparse checkpoint tensor shape/dtype differs: " + key)
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != self._immutable_backbone_keys() or result.unexpected_keys:
            raise ValueError("Sparse checkpoint key validation failed")

    def forward(self, x, task=None, *, processed_sizes=None, return_aux=False):
        if task not in (None, "cod", "iml") or x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point():
            raise ValueError("Expected BCHW floating RGB and a known task")
        if processed_sizes is not None:
            checked_sizes(x, processed_sizes)
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        image = (x - mean) / std
        outputs = {}
        for t in ("cod", "iml"):
            if task not in (None, t):
                continue
            features = self.encoder(image, t, processed_sizes=processed_sizes)
            outputs[t] = getattr(self, t + "_decoder")(features, x.shape[-2:])
        return outputs

    def parameter_groups(self):
        private = {t + "_private": list(getattr(self, t + "_decoder").parameters())
                   + list(self.encoder.private_tunes[t].parameters()) for t in ["cod", "iml"]}
        if hasattr(self.encoder, "frequency_gates"):
            for t in ("cod", "iml"):
                private[t + "_private"].extend(self.encoder.frequency_gates[t].parameters())
        identifiers = {id(p) for values in private.values() for p in values}
        shared = [p for p in self.parameters() if p.requires_grad and id(p) not in identifiers]
        return {"shared": shared, **private}
