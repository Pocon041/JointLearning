# Third-party sources and adaptation scope

## EVP and SegFormer

`cod_iml_evp_official_aligned/vendor_evp.py` contains classes extracted from the EVP authors' released SegFormer implementation. Its source header retains NVIDIA attribution and refers to the complete [NVIDIA Source Code License](cod_iml_evp_official_aligned/NVIDIA_LICENSE.txt) and [EVP repository BSD notice](cod_iml_evp_official_aligned/EVP_LICENSE.txt).

Source: <https://github.com/NiFangBaAGe/Explicit-Visual-Prompt/blob/latest_branch/models/mmseg/models/backbones/mix_transformer_evp.py>.

The extracted classes are Mlp, Attention, Block, OverlapPatchEmbed, DWConv and PromptGenerator. Standalone shims replace timm/mmcv dependencies, and the FFT mask is allocated on the input device. The wrapper aligns ImageNet normalization, decoder fusion initialization and reference image-size metadata. Details and the original source hash are in [source_provenance.json](cod_iml_evp_official_aligned/source_provenance.json).

The presence of a BSD notice for the upstream EVP repository does not override the NVIDIA terms on its SegFormer-derived source. The initial release does not relicense these files as MIT or Apache.

## AdaTask

Reference: <https://github.com/EnnengYang/AdaTask>, author formula source <https://raw.githubusercontent.com/EnnengYang/AdaTask/master/adatask.py>.

`cod_iml_evp_joint_v3/optimizer.py` implements task-specific Adam moments and summed adaptive proposals. Role-based learning rates, AdamW decay, missing-task handling, clipping and serialization are explicit project adaptations. Author source files themselves are not vendored here; URL/hash metadata and the executed formula comparison are retained in `docs/reference_*.json`.

## MTLoRA

Reference: <https://github.com/scale-lab/MTLoRA> and <https://arxiv.org/abs/2403.20320>.

Private stage exits adapt the shared-stream/private-readout principle and low-rank projections to EVP/MiT. This is not a reproduction of the full Swin model or its task suite. In particular, private FFN FC2 consumes private hidden activations, and the private branch is only exposed at each stage's last block. Author source files are not vendored; source hashes are retained for comparison provenance.

## Compatibility operators

The copied parent wrapper also imports `operators.py`, which contains independent equation-based implementations inspired by FBNet FACA and ObjectFormer. These optional operators and the retained historical bridge classes are **not activated by the released EVP/QKV configuration**. Compatibility dependencies remain so the copied source and regression tests need no scientific-code changes.

Dataset and pretrained-weight licenses remain with their respective distributors. No dataset or model binary is included in this source release.
