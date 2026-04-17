# TokaMind

TokaMind provides a multi-modal, token-based Transformer pipeline for scientific and industrial signals.

The repository is split into two layers:
- `src/mmt/`: dataset-agnostic core library (model, codecs, transforms, training loop) — usable standalone without any external dataset integration (see `src/mmt/examples/` for a self-contained toy example)
- `scripts_mast/`: FAIR/MAST integration layer (task configs, data wiring, entry scripts)

## 📝 Description
TokaMind implements a schema-flexible tokenization pipeline and a modular multi-modal Transformer with per-output adapters.

The code corresponds to the official implementation introduced in [TokaMind: A Multi-Modal Transformer Foundation Model for Tokamak Plasma Dynamics](https://arxiv.org/abs/2602.15084), evaluated against the [TokaMark benchmark](https://arxiv.org/abs/2602.10132).

[![MMT architecture](assets/mmt_architecture.png)](assets/mmt_architecture.pdf)
*Figure: Tokenization + model flow.* Windowed multimodal inputs and actuators are chunked and compressed by signal-specific codecs into tokens. Tokens are projected to a shared model dimension, processed by a Transformer backbone, and mapped to targets via modality heads and per-output adapters.

## 🔗 Companion Resources

| Resource | Link |
|---|---|
| TokaMind paper | [arXiv:2602.15084](https://arxiv.org/abs/2602.15084) |
| TokaMark paper | [arXiv:2602.10132](https://arxiv.org/abs/2602.10132) |
| TokaMind repository | [UKAEA-IBM-STFC-Fusion-FMs/tokamind](https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamind) |
| TokaMark repository | [UKAEA-IBM-STFC-Fusion-FMs/tokamark](https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark) |
| VAE-FAIRMAST repository | _coming soon_ |
| Pretrained models (HuggingFace) | [UKAEA-IBM-STFC](https://huggingface.co/UKAEA-IBM-STFC) |

## 📚 Documentation
- [Configuration Guide](docs/config_guide.md)
- [Configuration Reference](docs/config_reference.md)
- [DCT3D Tuning](docs/tuning_dct3d.md)
- [Checkpointing and Warmstart](docs/checkpointing_and_warmstart.md)
- [Evaluation](docs/evaluation.md)
- [Datasets](docs/datasets.md)
- [Transforms](docs/transforms.md)
- [Model Architecture](docs/model_architecture.md)
- [Model Flexibility](docs/model_flexibility.md)

## 🗂️ Repository Layout
```text
.
├── src/mmt/                           # Core package (dataset-agnostic, usable standalone)
│   ├── data/                          # signal specs, codecs, transforms, datasets
│   ├── models/                        # transformer model blocks
│   ├── train/                         # training loop
│   ├── eval/                          # decode and eval helpers
│   ├── examples/                      # self-contained toy training example (no FAIR/MAST required)
│   └── utils/                         # logging, seeds, config validation
├── scripts_mast/                      # FAIR/MAST integration
│   ├── run_pretrain.py
│   ├── run_finetune.py
│   ├── run_eval.py
│   ├── mast_utils/
│   │   ├── config/                    # config loading modules
│   │   └── ...
│   └── configs/
├── docs/                              # project documentation
└── runs/                              # output runs and checkpoints
```

## 📦 Installation

**Recommended Python: 3.11+**

For full MAST experiments, clone all repositories side-by-side in the same parent folder (steps 1–3 below). For standalone use, only step 1 is required.

Create and activate a conda environment first:

```bash
conda create -n tokamind-env python=3.14
conda activate tokamind-env
```

**For Windows users, install `wheels` and `setuptools`:**
```bash
pip install -U pip setuptools wheel
```

---

### 1) Install TokaMind

```bash
git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamind.git
cd tokamind
pip install -e .
```

> **Standalone use:** The core `src/mmt/` package works without any MAST/TokaMark integration. To verify your installation or explore the model independently, run the self-contained toy example:
> ```bash
> python src/mmt/examples/toy_train.py
> ```
> No benchmark data or external repositories required.

#### Developer setup (lint + format hooks)

For contributors, install dev dependencies and enable pre-commit hooks:

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files   # recommended once after setup
```

The pre-commit configuration runs `ruff check` and `ruff format`.

---

### 2) TokaMark integration

Required to run the MAST benchmark tasks via `scripts_mast/`.

```bash
git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark.git
cd tokamark
pip install -e .
```

---

### 3) VAE-FAIRMAST integration (optional)

*Coming soon.* Only needed to reproduce the VAE embedding experiments for Group-1.

```bash
git clone <vae-fairmast-repo-url>   # coming soon
cd vae-fairmast
pip install -e .
```

## 🤗 Pretrained Model

Pretrained TokaMind checkpoints (trained on MAST data) are available on HuggingFace:

- [tokamind-base](https://huggingface.co/UKAEA-IBM-STFC/tokamind-base)
- [tokamind-tiny](https://huggingface.co/UKAEA-IBM-STFC/tokamind-tiny)

The HuggingFace repository includes:
- Model weights (`checkpoints/best`)
- Embedding artifacts (`embeddings/dct3d.yaml`, `embeddings/dct3d_indices/*.npy`)
- Config snapshot used for pretraining

To use it, download and place the model under `runs/` so it matches the expected layout:

```
runs/
└── tokamind_base/
    ├── tokamind_base.yaml
    ├── checkpoints/
    │   └── best
    └── embeddings/
        ├── dct3d.yaml
        └── dct3d_indices/
```

You can then warmstart a finetune directly from it — see [Checkpointing and Warmstart](docs/checkpointing_and_warmstart.md).

## 🚀 Run Workflow
### 1) Pretrain
```bash
python scripts_mast/run_pretrain.py \
  --task pretrain_inputs_actuators_to_inputs_outputs \
  --emb_profile dct3d \
  --run-id tokamind_base
```

### 2) Finetune
**Warmstart:**
```bash
python scripts_mast/run_finetune.py \
  --task task_2-1 \
  --init warmstart \
  --model tokamind_base \
  --emb_profile dct3d \
  --tag exp1
```

**Scratch:**
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

## ⚙️ Configuration Model
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

## 🧩 Embedding Resolution
DCT3D tuning is integrated in the training scripts and controlled through `embeddings.tune_embeddings`.

- Pretrain: `tune_embeddings.roles` selects which roles to tune.
- Finetune: `embeddings.mode` controls whether embeddings are inherited from source run (`source`) or read directly from config (`config`).
- Eval: embeddings are loaded from the evaluated training run.

Details are in [DCT3D Tuning](docs/tuning_dct3d.md).

## 📁 Outputs
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

## 📄 License
See [License file](LICENSE.md).

---

## Citing TokaMind

If you use TokaMind, please cite our work as:

    @article{boschi2026tokamind,
      title={TokaMind: A Multi-Modal Transformer Foundation Model for Tokamak Plasma Dynamics},
      author={
        Boschi, Tobia and Loreti, Andrea and Amorisco, Nicola C and Ordonez-Hurtado, Rodrigo H and
        Rousseau, C{\'e}cile and Holt, George K and Sz{\'e}kely, Eszter and Whittle, Alexander and
        Jackson, Samuel and Agnello, Adriano and Pamela, Stanislas and Pascale, Alessandra and
        Akers, Robert and Bernabe Moreno, Juan and Thorne, Sue and Zayats, Mykhaylo
      },
      journal={arXiv preprint arXiv:2602.15084},
      year={2026}
    }

If you use the TokaMark benchmark alongside TokaMind, please also cite:

    @article{rousseau2026tokamark,
      title={TokaMark: A Comprehensive Benchmark for MAST Tokamak Plasma Models},
      author={
        Rousseau, C{\'e}cile and Jackson, Samuel and Ordonez-Hurtado, Rodrigo H. and
        Amorisco, Nicola C. and Boschi, Tobia and Holt, George K and Loreti, Andrea and 
        Sz{\'e}kely, Eszter and Whittle, Alexander and Agnello, Adriano and Pamela, Stanislas and 
        Pascale, Alessandra and Akers, Robert and Bernabe Moreno, Juan and Thorne, Sue and 
        Zayats, Mykhaylo
      },
      journal={arXiv preprint arXiv:2602.10132},
      year={2026}
    }
