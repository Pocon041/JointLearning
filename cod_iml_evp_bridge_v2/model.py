"""Preserve EVP dual enhancement and add a spatially anchored relation library."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cod_iml_evp_official_aligned.model import JointEVPModel
from .modules import (EnergyBandPyramid, LocalFrequencyBridge, LocalRelationLibrary,
                      PrivateFrequencyReader, SignedBandBank)


class LocalBridgeEncoder(nn.Module):
    def __init__(self, base, *, band_radii, band_transition, energy_floor, frequency_enabled=True):
        super().__init__()
        for name, module in base.named_children():
            self.add_module(name, module)
        for name in ["channels", "depths", "freeze_backbone", "checkpoint_blocks", "hfc_enabled", "enhancement"]:
            setattr(self, name, getattr(base, name))
        widths = [c // self.prompt_generator.scale_factor for c in self.channels]
        self.band_bank = SignedBandBank(band_radii, band_transition)
        self.band_encoder = EnergyBandPyramid(widths, energy_floor)
        self.relations = nn.ModuleList(LocalRelationLibrary(c) for c in widths)
        self.local_bridges = nn.ModuleList(LocalFrequencyBridge(c) for c in widths)
        self.readers = nn.ModuleDict({t: nn.ModuleList(PrivateFrequencyReader(c) for c in widths)
                                     for t in ["cod", "iml"]})
        self.new_outputs = nn.ModuleList(nn.Conv2d(2 * c, c, 1, bias=False) for c in widths)
        for layer in self.new_outputs:
            nn.init.zeros_(layer.weight)
        self.new_path_enabled = True
        # hfc_enabled controls the ORIGINAL EVP HFC only. This explicit switch
        # disables ALL frequency-derived inputs for a valid RGB-only ablation.
        self.frequency_enabled = bool(frequency_enabled)

    def _frequency_features(self, image, *, processed_sizes=None):
        if processed_sizes is not None:
            raise ValueError("processed sizes require the full-image encoder")
        bands = self.band_encoder(self.band_bank(image)) if self.new_path_enabled and self.frequency_enabled else None
        hfc = self.prompt_generator.init_handcrafted(image) if self.frequency_enabled and self.hfc_enabled else None
        return bands, hfc

    def forward(self, image, task, *, return_aux=False, processed_sizes=None):
        bands, hfc = self._frequency_features(image, processed_sizes=processed_sizes)
        x, outputs, auxiliary = image, [], []
        for s in range(1, 5):
            x, h, w = getattr(self.backbone, f"patch_embed{s}")(x)
            pe = getattr(self.prompt_generator, f"embedding_generator{s}")(x)
            frequency_tokens = hfc[s - 1] if hfc is not None else torch.zeros_like(pe)
            base = pe + frequency_tokens
            content = pe.transpose(1, 2).reshape(image.shape[0], -1, h, w)
            frequency = frequency_tokens.transpose(1, 2).reshape_as(content)
            if self.enhancement != "evp":
                structure = self.structure[s - 1](content + frequency)
                trace = self.trace[s - 1](content, frequency)
                ds, dt = structure - frequency, trace - frequency
                gs, gt = self.task_gates[task][s - 1](torch.cat((content, frequency, ds, dt), 1)).sigmoid().chunk(2, 1)
                base = base + (gs * ds + gt * dt).flatten(2).transpose(1, 2)
            if bands is not None:
                evidence, query = self.readers[task][s - 1].prepare(bands[s - 1], content)
                relation = self.relations[s - 1](bands[s - 1], content)
                result = self.local_bridges[s - 1](relation, query, return_aux=return_aux)
                message, details = result if return_aux else (result, {})
                combined, gate = self.readers[task][s - 1].finish(evidence, content, message)
                correction = self.new_outputs[s - 1](combined)
                base = base + correction.flatten(2).transpose(1, 2)
                if return_aux:
                    details.update(scale=s, gate_mean=gate.detach().mean(),
                                   correction_rms=correction.detach().float().square().mean().sqrt(),
                                   band_rms=bands[s - 1].detach().float().square().mean().sqrt())
                    auxiliary.append(details)
            up = getattr(self.prompt_generator, f"shared_mlp{s}")
            for i, block in enumerate(getattr(self.backbone, f"block{s}")):
                prompt = up(self.task_tunes[task][s - 1][i](base))
                if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                    x = checkpoint(block, x + prompt, h, w, use_reentrant=False, preserve_rng_state=True)
                else:
                    x = block(x + prompt, h, w)
            x = getattr(self.backbone, f"norm{s}")(x).transpose(1, 2).reshape(image.shape[0], -1, h, w)
            outputs.append(x)
        return (tuple(outputs), auxiliary) if return_aux else tuple(outputs)


class JointEVPBridgeV2Model(JointEVPModel):
    encoder_type = LocalBridgeEncoder

    def __init__(self, *, enhancement="local_bridge", baseline_enhancement="dual",
                 band_radii=(.25, .65), band_transition=.08, energy_floor=.1,
                 frequency_enabled=True, **base_config):
        if enhancement != "local_bridge" or baseline_enhancement not in ["evp", "dual"]:
            raise ValueError("Expected local_bridge with evp/dual baseline")
        super().__init__(enhancement=baseline_enhancement, **base_config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed() + 8001)
            self.encoder = self.encoder_type(self.encoder, band_radii=band_radii,
                band_transition=band_transition, energy_floor=energy_floor,
                frequency_enabled=frequency_enabled)
        self.config.update(enhancement=enhancement, baseline_enhancement=baseline_enhancement,
                           band_radii=tuple(band_radii), band_transition=band_transition, energy_floor=energy_floor,
                           frequency_enabled=bool(frequency_enabled))

    def forward(self, x, task=None, *, return_aux=False, processed_sizes=None):
        if task not in [None, "cod", "iml"] or x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point():
            raise ValueError("Expected BCHW floating RGB and a known task")
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        image = (x - mean) / std
        out, auxiliary = {}, {}
        for t in ["cod", "iml"]:
            if task not in [None, t]:
                continue
            result = self.encoder(image, t, return_aux=return_aux, processed_sizes=processed_sizes)
            features, details = result if return_aux else (result, None)
            out[t] = getattr(self, t + "_decoder")(features, x.shape[-2:])
            if return_aux:
                auxiliary[t] = details
        if return_aux:
            out["aux"] = auxiliary
        return out

    def parameter_groups(self):
        roles = super().parameter_groups()
        extra = {t: list(self.encoder.readers[t].parameters()) for t in ["cod", "iml"]}
        private_ids = {id(p) for values in extra.values() for p in values}
        roles["shared"] = [p for p in roles["shared"] if id(p) not in private_ids]
        for task in ["cod", "iml"]:
            roles[task + "_private"].extend(extra[task])
        return roles
