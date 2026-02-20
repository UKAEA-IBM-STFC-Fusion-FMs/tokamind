# TokaMind

TokaMind provides a multi-modal, token-based Transformer pipeline for scientific and industrial signals.

The repository is split into two layers:
- `src/mmt/`: dataset-agnostic core library (model, codecs, transforms, training loop)
- `scripts_mast/`: FAIR/MAST integration layer (task configs, data wiring, entry scripts)

## Description
TokaMind implements a schema-flexible tokenization pipeline and a modular multi-modal Transformer with per-output adapters.

[![MMT architecture](assets/mmt_architecture.png)](assets/mmt_architecture.pdf)
*Figure: Tokenization + model flow.* Windowed multimodal inputs and actuators are chunked and compressed by signal-specific codecs into tokens. Tokens are projected to a shared model dimension, processed by a Transformer backbone, and mapped to targets via modality heads and per-output adapters.

## Documentation
- [Configuration Guide](docs/config_guide.md)
- [Configuration Reference](docs/config_reference.md)
- [DCT3D Tuning](docs/tuning_dct3d.md)
- [Checkpointing and Warmstart](docs/checkpointing_and_warmstart.md)
- [Evaluation](docs/evaluation.md)
- [Datasets](docs/datasets.md)
- [Transforms](docs/transforms.md)
- [Model Architecture](docs/model_architecture.md)
- [Model Flexibility](docs/model_flexibility.md)

## Repository Layout
```text
.
├── src/mmt/                           # Core package
│   ├── data/                          # signal specs, codecs, transforms, datasets
│   ├── models/                        # transformer model blocks
│   ├── train/                         # training loop
│   ├── eval/                          # decode and eval helpers
│   └── utils/                         # logging, seeds, config validation
├── scripts_mast/                      # FAIR/MAST integration
│   ├── run_pretrain.py
│   ├── run_finetune.py
│   ├── run_eval.py
│   ├── mast_utils/
│   │   ├── config/                  # config loading modules
│   │   └── ...
│   └── configs/
├── docs/                              # project documentation
└── runs/                              # output runs and checkpoints
```

## 📦 Installation

This submission consists of up to three local repositories (we suggest to leave them side-by-side in the same parent folder):

- `fairmast-data-preprocessing/` (TokaMark benchmark + data utilities)
- `tokamind/` (TokaMind framework)
- `vae-fairmast/` (optional: VAE embeddings used for Group-1 experiments)

### 1) Create and activate a conda environment

> **TO UPDATE**: add git folder address of benchmark and VAE when available

**Recommended Python:** **3.11+**
```bash
conda create -n tokamind-env python=3.14
conda activate tokamind-env
```

**For Windows users, install `wheels` and `setuptools`:**
```bash
pip install -U pip setuptools wheel
```

### 2) Install TokaMind

```bash
cd ../tokamind
pip install -e .
# pip install -e ".[dev]"
```

### 3) Install the benchmark/data package

Benchmark integration to run Script MAST

```bash
cd fairmast-data-preprocessing
pip install -e .
```

### 4) (Optional) Install VAE embeddings support

Only needed to reproduce the VAE embedding experiments for Group-1.

```bash
cd ../vae-fairmast
pip install -e .
```

## Run Workflow
### 1) Pretrain
```bash
python scripts_mast/run_pretrain.py \
  --task pretrain_inputs_actuators_to_inputs_outputs \
  --emb_profile dct3d \
  --run-id tokamind_base
```

### 2) Finetune
Warmstart:
```bash
python scripts_mast/run_finetune.py \
  --task task_2-1 \
  --init warmstart \
  --model tokamind_base \
  --emb_profile dct3d \
  --tag exp1
```

Scratch:
```bash
python scripts_mast/run_finetune.py \
  --task task_2-1 \
  --init scratch \
  --emb_profile dct3d \
  --tag exp1
```

### 3) Evaluate
```bash
python scripts_mast/run_eval.py \
  --task task_2-1 \
  --model ft-task_2-1-ws-tokamind_base-exp1
```

## Configuration Model
Configuration is convention-based and merged by phase.

Base files:
- `scripts_mast/configs/common/embeddings.yaml`
- `scripts_mast/configs/common/pretrain.yaml`
- `scripts_mast/configs/common/finetune.yaml`
- `scripts_mast/configs/common/eval.yaml`

Task files:
- `scripts_mast/configs/tasks_overrides/<task>/<phase>_overrides.yaml` (optional)
- `scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/<profile>.yaml`

Finetune model keys in `scripts_mast/configs/common/finetune.yaml`:
- `model_scratch`: scratch-only base architecture
- `finetune_model_overrides`: model overrides applied in both scratch and warmstart
- `warmstart.model_overrides`: warmstart-only model overrides

Details are in:
- [Configuration Guide](docs/config_guide.md)
- [Configuration Reference](docs/config_reference.md)

## Embedding Resolution
DCT3D tuning is integrated in the training scripts and controlled through `embeddings.tune_embeddings`.

- Pretrain: `tune_embeddings.roles` selects which roles to tune.
- Finetune: `embeddings.mode` controls whether embeddings are inherited from source run (`source`) or read directly from config (`config`).
- Eval: embeddings are loaded from the evaluated training run.

Details are in [DCT3D Tuning](docs/tuning_dct3d.md).

## Outputs
Training runs are written under:
- `runs/<run_id>/`

Evaluation runs are written under:
- `runs/<model_id>/eval/`

Each training run stores:
- config snapshot (`<run_id>.yaml`)
- checkpoints (`checkpoints/best` and `checkpoints/latest`)
- embedding artifacts (`embeddings/dct3d.yaml`, `embeddings/dct3d_indices/*.npy` when rank mode is used)

See:
- [Checkpointing and Warmstart](docs/checkpointing_and_warmstart.md)
- [Evaluation](docs/evaluation.md)

## License
MIT (see `pyproject.toml`).
