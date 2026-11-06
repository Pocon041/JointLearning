"""Task-specific consumption of shared HFC evidence, with an exact EVP start."""
import torch
from torch import nn


class FrequencyGate(nn.Module):
    """Bounded, channel-wise gate before RGB/HFC addition; no new normalization.

    The same gate form is used for COD and IML, with independent parameters.
    2*sigmoid(0)=1 makes initialization exactly equal to EVP's PE+HFC. A zero
    frequency tensor contributes exactly zero, regardless of the RGB content.
    The proposed benefit is unverified; this is not a transfer guarantee.
    """
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Linear(2 * channels, channels)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, content, frequency):
        if content.shape != frequency.shape:
            raise ValueError("content and frequency token shapes must agree")
        gate = 2 * self.projection(torch.cat((content, frequency), dim=-1)).sigmoid()
        return content + gate * frequency
