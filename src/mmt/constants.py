# Token roles
ROLE_CONTEXT = 0
ROLE_ACTUATOR = 1
ROLE_OUTPUT = 2

# Explicit PAD semantics for token-level fields
PAD_ID = -1  # never a real signal_id
PAD_ROLE = -1  # no semantic role
PAD_MOD = -1  # no modality
PAD_POS = 0  # arbitrary, unused by model for PAD
