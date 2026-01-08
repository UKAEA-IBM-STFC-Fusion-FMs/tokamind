# MMT: Multi-Modal Transformer

MMT is a **multi-modal, token-based Transformer** designed for scientific / industrial sensor data (e.g. time-series, profiles, video) with a **clean separation** between:

- **`src/mmt/`**: the *dataset-agnostic* core library (model + data pipeline primitives)
- **`scripts_mast/`**: the FAIR/MAST integration layer (task configs, dataset wiring, training/eval scripts)

If you’re new to the repo, start with the **toy example** in `examples/` (runs on synthetic data and does not require the baseline dataset stack), then move to `scripts_mast/` for real tasks.

---

## Description

The core idea is to represent heterogeneous modalities as a **sequence of tokens**, where each token corresponds to a compressed representation of a chunk of data (e.g., a short time segment for time-series). Tokens from inputs/actuators are processed by a Transformer backbone, and outputs are produced by lightweight **per-output adapters**.

Key features:

- **Dataset-agnostic core (`mmt/`)**: the model and token pipeline are reusable across domains.
- **Convention-based configuration**: common defaults + per-task overrides, with phases for `pretrain`, `finetune`, `eval`, and `tune_dct3d`.
- **Per-task embedding tuning**: `run_tune_dct3d.py` writes `embeddings_overrides.yaml` inside the task folder.
- **Flexible training/evaluation**: warm-start vs resume, forced-drop ablations at eval time, cached vs streamed datasets.

For deeper details, see:
- `docs/model_architecture.md`
- `docs/model_flexibility.md`

---

## Visuals

Architecture and pipeline diagrams live in:
- `docs/model_architecture.md`
- `docs/transforms.md`

(You can add figures/screenshots here later if desired.)

---

## 📦 Installation

There are two common workflows:

1) **Core install + toy example** (recommended first; no baseline required)  
2) **Full MAST integration** (requires the baseline repository and datasets)

### 1) Install MMT (core library)

Clone and install in editable mode:

```bash
git clone https://github.com/<org>/multi-modal-transformer.git
cd multi-modal-transformer

python -m pip install -U pip
pip install -e .
# Optional developer extras:
pip install -e ".[dev]"
```

Smoke test with synthetic data:

```bash
python examples/toy_train.py --config examples/configs/toy.yaml
```

### 2) Full MAST integration (baseline repository)

This repository is designed to run on top of a **Baseline Environment** (FAIR/MAST preprocessing + datasets).

Follow these steps:

#### a) Install the baseline repository

```bash
git clone https://github.com/<org>/<baseline-repo>.git
cd <baseline-repo>

# complete block (dataset + deps)
```

Make sure the baseline repo is importable in the same Python environment used by MMT
(e.g., via `pip install -e .` in the baseline repo, or by setting `PYTHONPATH`).

#### b) Install MMT

```bash
git clone https://github.com/<org>/multi-modal-transformer.git
cd multi-modal-transformer

pip install -e .
pip install -e ".[dev]"
```

---

## Usage

### 1) Baseline-free toy example (synthetic data)

Runs a tiny training loop on synthetic data to demonstrate the core APIs:

```bash
python examples/toy_train.py --config examples/configs/toy.yaml
```

### 2) Run training/evaluation with MAST integration

All phase scripts use the same pattern: pass a **task folder name** under
`scripts_mast/configs/tasks/<task>/`.

Finetune:

```bash
python scripts_mast/run_finetune.py --task task_2-1
```

Pretrain (example):

```bash
python scripts_mast/run_pretrain.py --task pretrain_inputs_actuators_to_inputs_outputs
```

Evaluate:

1. In `scripts_mast/configs/tasks/<task>/eval_overrides.yaml`, set:

   ```yaml
   model_source:
     run_dir: "runs/<training_run_id>"
   ```

2. Run:

   ```bash
   python scripts_mast/run_eval.py --task task_2-1
   ```

Tune embedding parameters (DCT3D) for a task:

```bash
python scripts_mast/run_tune_dct3d.py --task task_2-1
```

This writes:

```
scripts_mast/configs/tasks/<task>/embeddings_overrides.yaml
```

### 3) Configuration

Configuration is **convention-based** (no pointers inside YAML). The loader merges:

1) `scripts_mast/configs/common/core.yaml`  
2) `scripts_mast/configs/common/embeddings.yaml`  
3) `scripts_mast/configs/common/<phase>.yaml`  
4) `scripts_mast/configs/tasks/<task>/task.yaml`  
5) `scripts_mast/configs/tasks/<task>/<phase>_overrides.yaml` *(optional)*  
6) `scripts_mast/configs/tasks/<task>/embeddings_overrides.yaml` *(optional)*

See:
- `docs/config_guide.md`

---

## Documentation

Recommended reading order:

- `docs/config_guide.md` — config structure, merge order, phases, and run directories
- `docs/model_architecture.md` — model blocks and data flow
- `docs/model_flexibility.md` — warm-start/resume, finetune/eval flexibility
- `docs/datasets.md` — cached vs streamed datasets, epoch semantics
- `docs/transforms.md` — transforms pipeline and window dict contract
- `docs/checkpointing_and_warmstart.md` — checkpoints, overlap loading, model parts
- `docs/evaluation.md` — metrics/traces, forced-drop ablations, eval outputs
- `docs/tuning_embeddings.md` — DCT3D tuning and `embeddings_overrides.yaml`

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
