"""Backward-compatible psi error-map interface; shared implementation in error_map."""

from .error_map import FieldErrorMapMetric
from .grid import GSEvalGrid


class PsiErrorMapMetric(FieldErrorMapMetric):
    """Preserve the existing psi call signature and NPZ/summary names."""

    def __init__(self, *, grid: GSEvalGrid) -> None:
        super().__init__(grid=grid, signal="psi")

    def add_batch(self, *, psi_native, psi_gt_native, window_mask) -> None:
        super().add_batch(pred_native=psi_native, true_native=psi_gt_native, window_mask=window_mask)
