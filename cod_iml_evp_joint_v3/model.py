"""Shared Q/KV plus private per-stage readouts, adapted to original EVP.

Borrowed from MTLoRA: shared stream continues into deeper stages; private stage
outputs feed only the labeled decoder. Shared Q/KV remains unchanged. Optional
private low-rank corrections affect attention output projection and FFN FC1/FC2
in the final block of each stage. No Swin/PASCAL reproduction claim is made.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from cod_iml_evp_reference.model import ReferenceEVPModel
from cod_iml_evp_full_image.geometry import handcrafted_features_before_padding


class LowRankDelta(nn.Module):
    def __init__(self, layer, rank, alpha):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank <= min(layer.weight.shape):
            raise ValueError("invalid private exit rank")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("invalid private exit scale")
        self.lora_A = nn.Parameter(layer.weight.new_empty(rank, layer.weight.shape[1]))
        self.lora_B = nn.Parameter(layer.weight.new_zeros(layer.weight.shape[0], rank))
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x=None):
        if x is None:
            return self.scale * (self.lora_B @ self.lora_A)
        return self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)


class StageExit(nn.Module):
    # This readout does not add task-specific QKV. The common attention relation
    # and generic weights remain available to both tasks, as in the author code.
    PROJECTIONS = (("attn_proj", "attn.proj"), ("fc1", "mlp.fc1"), ("fc2", "mlp.fc2"))

    def __init__(self, block, rank, alpha):
        super().__init__()
        for name, path in self.PROJECTIONS:
            layer = block.get_submodule(path)
            if not isinstance(layer, nn.Linear) or layer.weight.requires_grad:
                raise ValueError("private exit requires frozen plain projection/FFN matrices")
            self.add_module(name, LowRankDelta(layer, rank, alpha))

    def forward(self, block, tokens, h, w):
        # Same MiT attention/FFN operations as vendor_evp.Block, with additive
        # low-rank outputs at the three declared projections. Do NOT merge a tiny
        # delta into W before BF16 autocast: rounding can erase it completely.
        x = block.norm1(tokens)
        attn = block.attn
        b, n, c = x.shape
        q = attn.q(x).reshape(b, n, attn.num_heads, c // attn.num_heads).permute(0, 2, 1, 3)
        if attn.sr_ratio > 1:
            reduced = x.permute(0, 2, 1).reshape(b, c, h, w)
            reduced = attn.norm(attn.sr(reduced).reshape(b, c, -1).permute(0, 2, 1))
        else:
            reduced = x
        kv = attn.kv(reduced).reshape(b, -1, 2, attn.num_heads, c // attn.num_heads).permute(2, 0, 3, 1, 4)
        probability = attn.attn_drop(((q @ kv[0].transpose(-2, -1)) * attn.scale).softmax(dim=-1))
        context = (probability @ kv[1]).transpose(1, 2).reshape(b, n, c)
        attention_output = attn.proj_drop(attn.proj(context) + self.attn_proj(context))
        x = tokens + block.drop_path(attention_output)
        normalized = block.norm2(x)
        mlp = block.mlp
        hidden = mlp.fc1(normalized) + self.fc1(normalized)
        hidden = mlp.drop(mlp.act(mlp.dwconv(hidden, h, w)))
        output = mlp.drop(mlp.fc2(hidden) + self.fc2(hidden))
        return x + block.drop_path(output)


class JointStrategyModel(ReferenceEVPModel):
    def __init__(self, *, private_stage_exits=False, exit_rank=4, exit_alpha=4., **config):
        super().__init__(**config)
        self.config.update(private_stage_exits=bool(private_stage_exits), exit_rank=exit_rank, exit_alpha=exit_alpha)
        if private_stage_exits:
            if self.encoder.frequency_fusion != "sum" or self.encoder.sharing != "all_private":
                raise ValueError("stage-exit experiments retain original EVP prompt sharing/fusion")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() + 20001)
                self.encoder.exit_readouts = nn.ModuleDict({task: nn.ModuleList(
                    StageExit(getattr(self.encoder.backbone, f"block{s}")[-1], exit_rank, exit_alpha)
                    for s in range(1, 5)) for task in ("cod", "iml")})

    def parameter_groups(self):
        roles = super().parameter_groups()
        if hasattr(self.encoder, "exit_readouts"):
            private = {task: list(self.encoder.exit_readouts[task].parameters()) for task in ("cod", "iml")}
            ids = {id(p) for values in private.values() for p in values}
            roles["shared"] = [p for p in roles["shared"] if id(p) not in ids]
            for task in private:
                roles[task + "_private"].extend(private[task])
        return roles

    def _features(self, image, task, processed_sizes=None):
        encoder = self.encoder
        hfc = None
        if encoder.frequency_enabled and encoder.hfc_enabled:
            hfc = (encoder.prompt_generator.init_handcrafted(image) if processed_sizes is None else
                   handcrafted_features_before_padding(encoder.prompt_generator, image, processed_sizes))
        x, outputs = image, []
        for s in range(1, 5):
            x, h, w = getattr(encoder.backbone, f"patch_embed{s}")(x)
            content = getattr(encoder.prompt_generator, f"embedding_generator{s}")(x)
            base = content + hfc[s - 1] if hfc is not None else content
            up = getattr(encoder.prompt_generator, f"shared_mlp{s}")
            blocks = getattr(encoder.backbone, f"block{s}")
            for i, block in enumerate(blocks):
                prompt = up(encoder.private_tunes[task][s - 1][i](base))
                tokens = x + prompt
                if i == len(blocks) - 1:
                    readout = encoder.exit_readouts[task][s - 1]
                    devices = [image.device.index or 0] if image.device.type == "cuda" else []
                    # Couple stochastic masks to the shared exit. With zero
                    # private deltas the readout is exactly its shared parent,
                    # and adding it consumes no extra global RNG draws.
                    with torch.random.fork_rng(devices=devices):
                        if encoder.checkpoint_blocks and self.training and torch.is_grad_enabled():
                            # Bind block/readout as arguments: no late-bound loop closure.
                            private = checkpoint(readout, block, tokens, h, w, use_reentrant=False, preserve_rng_state=True)
                        else:
                            private = readout(block, tokens, h, w)
                if encoder.checkpoint_blocks and self.training and torch.is_grad_enabled():
                    x = checkpoint(block, tokens, h, w, use_reentrant=False, preserve_rng_state=True)
                else:
                    x = block(tokens, h, w)
            norm = getattr(encoder.backbone, f"norm{s}")
            # Export the private readout; propagate only the shared result.
            outputs.append(norm(private).transpose(1, 2).reshape(image.shape[0], -1, h, w))
            x = norm(x).transpose(1, 2).reshape(image.shape[0], -1, h, w)
        return tuple(outputs)

    def forward(self, image, task=None, *, processed_sizes=None, return_aux=False):
        if not hasattr(self.encoder, "exit_readouts"):
            return super().forward(image, task=task, processed_sizes=processed_sizes, return_aux=return_aux)
        if task not in (None, "cod", "iml") or image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("expected floating RGB and a known task")
        if processed_sizes is not None:
            from cod_iml_evp_full_image.geometry import checked_sizes
            checked_sizes(image, processed_sizes)
        normalized = (image - image.new_tensor([.485, .456, .406])[None, :, None, None]) / image.new_tensor([.229, .224, .225])[None, :, None, None]
        outputs = {}
        for t in ("cod", "iml"):
            if task not in (None, t):
                continue
            features = self._features(normalized, t, processed_sizes)
            outputs[t] = getattr(self, t + "_decoder")(features, image.shape[-2:])
        return outputs
