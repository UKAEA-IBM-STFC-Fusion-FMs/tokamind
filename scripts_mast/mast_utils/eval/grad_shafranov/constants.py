"""Physical and asset constants used only by Grad-Shafranov evaluation diagnostics.

The grid asset normally supplies ``mu0``. ``MU0`` is retained as the physical
fallback for assets that do not. Keeping these definitions here avoids coupling
evaluation diagnostics to the training-loss implementation.
"""

DEFAULT_GS_PARAMS_FILE = "scripts_mast/assets/grad_shafranov/grad_shafranov_params_mast_coo.npz"
MU0 = 1.2566370614359173e-06
PLASMA_JTOR_THRESHOLD = 1e-6
