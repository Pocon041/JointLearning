"""A cold-start EVP baseline plus protected private evidence and a shared bridge."""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cod_iml_evp_official_aligned.model import JointEVPModel
from .modules import (TASKS, BandPyramid, LocalBandRelations, SharedRelationBridge,
                      SignedBandBank, TaskReader)


class BridgePromptEncoder(nn.Module):
    def __init__(self, base, *, bridge_latents, bridge_pool_size, band_radii,
                 band_transition, cross_cod_to_iml, cross_iml_to_cod):
        super().__init__()
        # Move the actual author-derived EVP components without reinitializing.
        self.backbone = base.backbone
        self.prompt_generator = base.prompt_generator
        self.task_tunes = base.task_tunes
        self.channels, self.depths = base.channels, base.depths
        self.freeze_backbone = base.freeze_backbone
        self.checkpoint_blocks = base.checkpoint_blocks
        self.hfc_enabled = base.hfc_enabled
        self.cross_cod_to_iml = bool(cross_cod_to_iml)
        self.cross_iml_to_cod = bool(cross_iml_to_cod)
        widths = [c // self.prompt_generator.scale_factor for c in self.channels]
        self.band_bank = SignedBandBank(band_radii, band_transition)
        self.shared_band_encoder = BandPyramid(3, widths)
        self.private_bands = nn.ModuleDict({"cod": BandPyramid(9, widths)})
        self.private_bands["iml"] = copy.deepcopy(self.private_bands["cod"])
        self.relations = nn.ModuleList(LocalBandRelations(c) for c in widths)
        self.bridges = nn.ModuleList(SharedRelationBridge(c, bridge_latents, bridge_pool_size) for c in widths)
        self.readers = nn.ModuleDict({t: nn.ModuleList(TaskReader(c) for c in widths) for t in TASKS})
        self.private_outputs = nn.ModuleDict({t: nn.ModuleList(nn.Conv2d(c, c, 1) for c in widths) for t in TASKS})
        self.shared_outputs = nn.ModuleList(nn.Conv2d(c, c, 1) for c in widths)
        # Exact EVP behavior at initialization. The output projections learn on
        # update one; upstream new paths receive gradients from update two onward.
        for module in [self.shared_outputs, *self.private_outputs.values()]:
            for layer in module:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, image, task, *, return_aux=False):
        bands = self.band_bank(image)
        b, k, rgb, height, width = bands.shape
        shared_bands = self.shared_band_encoder(bands.reshape(b * k, rgb, height, width))
        all_bands = bands.flatten(1, 2)
        private = {t: self.private_bands[t](all_bands) for t in TASKS}
        hfc = self.prompt_generator.init_handcrafted(image)
        if not self.hfc_enabled:
            hfc = tuple(torch.zeros_like(v) for v in hfc)
        cross_enabled = self.cross_iml_to_cod if task == "cod" else self.cross_cod_to_iml
        x, features, auxiliary = image, [], []
        for s in range(1, 5):
            x, h, w = getattr(self.backbone, f"patch_embed{s}")(x)
            embedding = getattr(self.prompt_generator, f"embedding_generator{s}")(x)
            content = embedding.transpose(1, 2).reshape(b, -1, h, w)
            band_features = shared_bands[s - 1].reshape(b, k, -1, h, w)
            views = {t: private[t][s - 1] for t in TASKS}
            relation = self.relations[s - 1](band_features, content)
            hub = self.bridges[s - 1](relation, views, task, cross_enabled=cross_enabled)
            message, gate = self.readers[task][s - 1](views[task], content, hub)
            correction = (self.private_outputs[task][s - 1](views[task])
                          + self.shared_outputs[s - 1](message))
            base = embedding + hfc[s - 1] + correction.flatten(2).transpose(1, 2)
            up = getattr(self.prompt_generator, f"shared_mlp{s}")
            for i, block in enumerate(getattr(self.backbone, f"block{s}")):
                prompt = up(self.task_tunes[task][s - 1][i](base))
                if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                    x = checkpoint(block, x + prompt, h, w, use_reentrant=False, preserve_rng_state=True)
                else:
                    x = block(x + prompt, h, w)
            x = getattr(self.backbone, f"norm{s}")(x)
            x = x.transpose(1, 2).reshape(b, -1, h, w)
            features.append(x)
            if return_aux:
                auxiliary.append({"scale": s, "gate_mean": gate.detach().mean(),
                                  "private_rms": views[task].detach().float().square().mean().sqrt(),
                                  "relation_rms": relation.detach().float().square().mean().sqrt(),
                                  "correction_rms": correction.detach().float().square().mean().sqrt(),
                                  "cross_enabled": cross_enabled})
        return (tuple(features), auxiliary) if return_aux else tuple(features)


class JointEVPBridgeModel(JointEVPModel):
    def __init__(self, *, enhancement="bridge", bridge_latents=16, bridge_pool_size=8,
                 band_radii=(.25, .65), band_transition=.08,
                 cross_cod_to_iml=True, cross_iml_to_cod=True, **base_config):
        if enhancement != "bridge":
            raise ValueError("This model implements enhancement='bridge'")
        # Heads/base prompts are initialized exactly as in the frozen EVP model.
        super().__init__(enhancement="evp", **base_config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed() + 4001)
            self.encoder = BridgePromptEncoder(
                self.encoder, bridge_latents=bridge_latents, bridge_pool_size=bridge_pool_size,
                band_radii=band_radii, band_transition=band_transition,
                cross_cod_to_iml=cross_cod_to_iml, cross_iml_to_cod=cross_iml_to_cod)
        self.config.update(enhancement=enhancement, bridge_latents=bridge_latents,
                           bridge_pool_size=bridge_pool_size, band_radii=tuple(band_radii),
                           band_transition=band_transition,
                           cross_cod_to_iml=bool(cross_cod_to_iml),
                           cross_iml_to_cod=bool(cross_iml_to_cod))

    def forward(self, x, task=None, *, return_aux=False):
        if task not in (None, *TASKS):
            raise ValueError("Unknown task")
        if x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point():
            raise ValueError("Expected floating point BCHW RGB")
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        image = (x - mean) / std
        output, auxiliary = {}, {}
        for t in TASKS:
            if task not in (None, t):
                continue
            result = self.encoder(image, t, return_aux=return_aux)
            features, details = result if return_aux else (result, None)
            output[t] = getattr(self, t + "_decoder")(features, x.shape[-2:])
            if return_aux:
                auxiliary[t] = details
        if return_aux:
            output["aux"] = auxiliary
        return output

    def parameter_groups(self):
        roles = {}
        for task in TASKS:
            modules = [getattr(self, task + "_decoder"), self.encoder.task_tunes[task],
                       self.encoder.private_bands[task], self.encoder.private_outputs[task],
                       self.encoder.readers[task]]
            roles[task + "_private"] = [p for module in modules for p in module.parameters() if p.requires_grad]
        private_ids = {id(p) for group in roles.values() for p in group}
        shared = [p for p in self.parameters() if p.requires_grad and id(p) not in private_ids]
        return {"shared": shared, **roles}
