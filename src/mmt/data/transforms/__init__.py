"""
Data transformation pipeline for MMT.

This package provides composable transform components for processing
multi-modal time-series data through the MMT pipeline.

Key modules
-----------
- chunk_windows.py          : split windows into fixed-size chunks
- trim_chunks.py            : trim chunks to valid lengths
- select_valid_windows.py   : filter windows based on validity criteria
- embed_chunks.py           : apply codecs to compress/embed chunks
- build_tokens.py           : convert embedded chunks to model tokens
- finalize_window.py        : prepare final window batch format
- tune_ranked_dct3d.py      : DCT3D coefficient tuning transform
- compose.py                : compose multiple transforms into pipeline
"""
