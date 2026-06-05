"""
Embedding and codec utilities for MMT.

This package provides codec implementations for compressing and embedding time-series signals
into lower-dimensional representations, together with differentiable torch decoders for use in
training losses and eval.

Key modules
-----------
- base.py           : TorchDecoder abstract base class
- codec_utils.py    : codec and decoder factories, shape helpers
- dct3d.py          : DCT3D encoder (numpy) + DCT3DTorchDecoder
- identity.py       : Identity encoder (numpy) + IdentityTorchDecoder
- vae.py            : VAE encoder (numpy) + VAETorchDecoder
"""

# Keep this package init lightweight: codec modules have optional heavy deps.
