# Configuration Guide

This document explains how configuration works in the **multi-modal-transformer (MMT)** project and how to run an experiment using the YAML files.

The goal is to keep things simple for the user: **one top-level config file per run**, with a clear structure underneath.

---

## 1. Overview: three configuration layers

For each task (e.g. `task_2-1`) we use three logical layers:

1. **Experiment base**  
   `mmt/configs/task_2-1/experiment_base.yaml`

   - Defines:
     - task id (`task`)
     - link to the **baseline FAIRMAST task config** (`baseline_config`), i.e. the YAML file from the separate baseline repository that defines the raw task (inputs, outputs, windowing, etc.)
     - global settings (`global.seed`)
     - data options (`data.local`, `data.subset_of_shots`)
     - preprocess settings (`preprocess.valid_windows`, `preprocess.chunking`, `preprocess.cache`)
     - high-level model shape (`model.backbone`, `model.modality_heads`, `model.adapters`)


2. **Embedding config**  
   `mmt/configs/task_2-1/embeddings_default.yaml` (and `embeddings_tuned.yaml`)

   - Defines how each **role** and **modality** is embedded:
     - `embeddings.defaults` for `(role, modality)` pairs (input / actuator / output × timeseries / profile / video)
     - `embeddings.per_signal_overrides` for per-signal special cases (e.g. tuned DCT settings for `equilibrium-psi`)


3. **Phase config** (entry point)  
   Examples:
   - `mmt/configs/task_2-1/finetune_default.yaml`
   - `mmt/configs/task_2-1/eval_default.yaml`

   - Defines:
     - `phase` (`"finetune"` or `"eval"`)
     - which `experiment_base` and `embedding_config` to use
     - phase-specific sections like `train`, `evaluation`, `run`, etc.

The **phase config** is the only file you pass to the loader; it in turn points to the experiment base and embedding config.

