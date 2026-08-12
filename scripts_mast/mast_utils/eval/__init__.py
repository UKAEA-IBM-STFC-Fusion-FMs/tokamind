"""
MAST/tokamark-style evaluation for MMT (scripts layer).

This subpackage holds the benchmark evaluation orchestration and the MAST/tokamark-style CRPS aggregation/reporting.
These encode the benchmark's reporting convention (window -> shot -> signal -> task hierarchy, tokamark CSV schema,
NRMSE-matched normalization) and are therefore MAST-specific — distinct from the generic, reusable scoring rules in
``mmt.scoring.crps``.
"""

from .benchmark_eval import evaluate_benchmark_and_diagnostics


__all__ = ["evaluate_benchmark_and_diagnostics"]
