"""examples/toy_train.py

Minimal, benchmark-free MMT example.

This script demonstrates how to:
  1) Load a small YAML config (configs/examples/toy.yaml)
  2) Build a tiny SignalSpecRegistry (inputs/actuators/outputs)
  3) Create a synthetic *window-level* dataset that produces the dictionaries
     expected by `mmt.data.MMTCollate`
  4) Train a `mmt.models.MultiModalTransformer` for a few steps

The goal is not to achieve meaningful performance, but to provide a simple
"first run" entrypoint for open-source users and CI smoke tests.

Run
---
  python examples/toy_train.py --config configs/examples/toy.yaml

Notes
-----
- This example depends only on `mmt/` (no scripts_mast/, no FAIRMAST).
- We set all dropout probabilities to 0.0 in the toy config to avoid PAD/drop
  edge cases and keep behaviour deterministic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from mmt.data import MMTCollate, SignalSpec
from mmt.data.signal_spec import SignalSpecRegistry
from mmt.models import MultiModalTransformer
from mmt.utils import set_seed

# Role ids are defined by the package
from mmt.constants import ROLE_CONTEXT, ROLE_ACTUATOR

# Allow running from a fresh repo checkout without `pip install -e .`.
# If the repository has a `src/` layout, add it to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.is_dir() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ToyWindowDataset(Dataset):
    """Synthetic window-level dataset compatible with MMTCollate.

    Each item is a *single window dict* with:
      - token embeddings (emb_chunks) + token metadata arrays (pos/id/mod/role)
      - per-output target embeddings (output_emb)

    Targets are generated from token embeddings using a fixed random mapping:
      y = mean(tokens) @ W + b (+ noise)

    This creates a learnable relationship so the training loss should decrease.
    """

    def __init__(
        self,
        *,
        num_windows: int,
        positions: int,
        embedding_dim: int,
        input_specs: List[SignalSpec],
        actuator_specs: List[SignalSpec],
        output_specs: List[SignalSpec],
        mod_to_id: Mapping[str, int],
        seed: int,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_windows = int(num_windows)
        self.positions = int(positions)
        self.embedding_dim = int(embedding_dim)
        self.input_specs = list(input_specs)
        self.actuator_specs = list(actuator_specs)
        self.output_specs = list(output_specs)
        self.mod_to_id = dict(mod_to_id)
        self.seed = int(seed)
        self.noise_std = float(noise_std)

        # Fixed random mapping per output signal id
        rng = np.random.default_rng(self.seed)
        self._W: Dict[int, np.ndarray] = {}
        self._b: Dict[int, np.ndarray] = {}
        for spec in self.output_specs:
            out_dim = int(spec.embedding_dim)
            self._W[spec.signal_id] = rng.standard_normal(
                size=(self.embedding_dim, out_dim)
            ).astype(np.float32)
            self._b[spec.signal_id] = rng.standard_normal(size=(out_dim,)).astype(
                np.float32
            )

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Deterministic sample-wise RNG (safe with DataLoader workers)
        rng = np.random.default_rng(self.seed + int(idx))

        emb_list: List[np.ndarray] = []
        pos_list: List[int] = []
        id_list: List[int] = []
        mod_list: List[int] = []
        role_list: List[int] = []
        name_list: List[str] = []

        # Build a fixed-length token sequence
        for p in range(self.positions):
            # inputs
            for spec in self.input_specs:
                emb = rng.standard_normal(size=(self.embedding_dim,)).astype(np.float32)
                emb_list.append(emb)
                pos_list.append(p)
                id_list.append(int(spec.signal_id))
                mod_list.append(int(self.mod_to_id[spec.modality]))
                role_list.append(int(ROLE_CONTEXT))
                name_list.append(spec.name)

            # actuators
            for spec in self.actuator_specs:
                emb = rng.standard_normal(size=(self.embedding_dim,)).astype(np.float32)
                emb_list.append(emb)
                pos_list.append(p)
                id_list.append(int(spec.signal_id))
                mod_list.append(int(self.mod_to_id[spec.modality]))
                role_list.append(int(ROLE_ACTUATOR))
                name_list.append(spec.name)

        # Targets: linear map of the mean token embedding
        x_mean = np.mean(np.stack(emb_list, axis=0), axis=0)  # (D,)

        output_emb: Dict[int, np.ndarray] = {}
        output_shapes: Dict[int, Any] = {}
        output_names: Dict[int, str] = {}

        for spec in self.output_specs:
            sid = int(spec.signal_id)
            y = x_mean @ self._W[sid] + self._b[sid]
            if self.noise_std > 0:
                y = y + rng.normal(scale=self.noise_std, size=y.shape).astype(
                    np.float32
                )
            output_emb[sid] = y.astype(np.float32).reshape(-1)
            output_shapes[sid] = tuple(output_emb[sid].shape)
            output_names[sid] = spec.name

        window: Dict[str, Any] = {
            "shot_id": int(idx),
            "window_index": int(idx),
            "emb_chunks": emb_list,
            "pos": np.asarray(pos_list, dtype=np.int32),
            "id": np.asarray(id_list, dtype=np.int32),
            "mod": np.asarray(mod_list, dtype=np.int16),
            "role": np.asarray(role_list, dtype=np.int8),
            "signal_name": np.asarray(name_list, dtype=object),
            "output_emb": output_emb,
            "output_shapes": output_shapes,
            "output_names": output_names,
        }
        return window


def _build_signal_specs(cfg: Mapping[str, Any]) -> SignalSpecRegistry:
    toy_cfg = cfg["toy"]
    modality = str(toy_cfg.get("modality", "timeseries"))
    D = int(toy_cfg["embedding_dim"])

    inputs = list(toy_cfg.get("inputs", []))
    actuators = list(toy_cfg.get("actuators", []))
    outputs = list(toy_cfg.get("outputs", []))

    specs: List[SignalSpec] = []
    sid = 0

    for name in inputs:
        specs.append(
            SignalSpec(
                name=str(name),
                role="input",
                modality=modality,
                encoder_name="identity",
                encoder_kwargs={},
                signal_id=sid,
                embedding_dim=D,
            )
        )
        sid += 1

    for name in actuators:
        specs.append(
            SignalSpec(
                name=str(name),
                role="actuator",
                modality=modality,
                encoder_name="identity",
                encoder_kwargs={},
                signal_id=sid,
                embedding_dim=D,
            )
        )
        sid += 1

    for name in outputs:
        specs.append(
            SignalSpec(
                name=str(name),
                role="output",
                modality=modality,
                encoder_name="identity",
                encoder_kwargs={},
                signal_id=sid,
                embedding_dim=D,
            )
        )
        sid += 1

    return SignalSpecRegistry(specs)


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch_to_device(
    batch: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    # token-level tensors
    for k in (
        "pos",
        "id",
        "mod",
        "role",
        "padding_mask",
        "input_mask",
        "actuator_mask",
    ):
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

    # outputs
    if "output_emb" in batch:
        batch["output_emb"] = {
            int(sid): t.to(device) for sid, t in batch["output_emb"].items()
        }
    if "output_mask" in batch:
        batch["output_mask"] = {
            int(sid): t.to(device) for sid, t in batch["output_mask"].items()
        }

    return batch


def _masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE averaged over unmasked samples.

    pred/target: (B, D)
    mask: (B,) boolean
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    # If nothing is masked out, regular MSE
    if bool(mask.all()):
        return F.mse_loss(pred, target)

    m = mask.float().unsqueeze(1)  # (B, 1)
    diff2 = (pred - target) ** 2
    denom = m.sum() * diff2.shape[1]
    return (diff2 * m).sum() / denom.clamp_min(1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MMT on synthetic data (toy example)."
    )
    p.add_argument(
        "--config",
        type=str,
        default="examples/configs/toy.yaml",
        help="Path to a YAML config (relative to repo root by default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve config path relative to repo root (examples/ is at repo_root/examples/)
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()

    cfg = _load_yaml(cfg_path)

    seed = int(cfg.get("seed", 0))
    set_seed(seed, deterministic=True, warn_only=True)

    device = _choose_device()
    print(f"[toy_train] device={device} | config={cfg_path}")

    # ------------------------------------------------------------------
    # Build signal specs + model
    # ------------------------------------------------------------------
    signal_specs = _build_signal_specs(cfg)

    model_cfg = cfg["model"]
    backbone = model_cfg["backbone"]

    model = MultiModalTransformer(
        signal_specs=signal_specs,
        d_model=int(backbone["d_model"]),
        n_layers=int(backbone["n_layers"]),
        n_heads=int(backbone["n_heads"]),
        dim_ff=int(backbone["dim_ff"]),
        dropout=float(backbone["dropout"]),
        max_positions=int(model_cfg.get("max_positions", 8)),
        modality_heads_cfg=model_cfg["modality_heads"],
        output_adapters_cfg=model_cfg["output_adapters"],
        backbone_activation=str(backbone.get("activation", "relu")),
        debug_tokens=False,
    ).to(device)

    # ------------------------------------------------------------------
    # Synthetic dataset + loader
    # ------------------------------------------------------------------
    toy_cfg = cfg["toy"]
    positions = int(toy_cfg["positions"])

    # Stable modality id mapping (same convention as BuildTokensTransform)
    mod_to_id = {m: i for i, m in enumerate(signal_specs.modalities)}

    input_specs = signal_specs.specs_for_role("input")
    actuator_specs = signal_specs.specs_for_role("actuator")
    output_specs = signal_specs.specs_for_role("output")

    target_cfg = toy_cfg.get("target") or {}
    noise_std = float(target_cfg.get("noise_std", 0.0))

    ds = ToyWindowDataset(
        num_windows=int(toy_cfg["num_windows"]),
        positions=positions,
        embedding_dim=int(toy_cfg["embedding_dim"]),
        input_specs=input_specs,
        actuator_specs=actuator_specs,
        output_specs=output_specs,
        mod_to_id=mod_to_id,
        seed=seed,
        noise_std=noise_std,
    )

    collate = MMTCollate(cfg.get("collate") or {})

    loader = DataLoader(
        ds,
        batch_size=int(toy_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Train (minimal loop)
    # ------------------------------------------------------------------
    lr = float(toy_cfg["lr"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    it = iter(loader)

    num_steps = int(toy_cfg["num_steps"])
    print_every = max(1, num_steps // 10)

    for step in range(1, num_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = _move_batch_to_device(batch, device)

        out = model(batch)
        preds = out["pred"]

        loss = torch.tensor(0.0, device=device)
        for sid, target in batch["output_emb"].items():
            sid_i = int(sid)
            pred = preds[sid_i]
            mask = batch["output_mask"][sid_i]
            loss = loss + _masked_mse(pred, target, mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % print_every == 0 or step == 1:
            print(f"step {step:4d}/{num_steps} | loss={loss.item():.6f}")

    # ------------------------------------------------------------------
    # Quick eval on one batch
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        batch = _move_batch_to_device(batch, device)
        out = model(batch)
        preds = out["pred"]

        loss = torch.tensor(0.0, device=device)
        for sid, target in batch["output_emb"].items():
            sid_i = int(sid)
            pred = preds[sid_i]
            mask = batch["output_mask"][sid_i]
            loss = loss + _masked_mse(pred, target, mask)

    print(f"[toy_train] done | eval_loss(one_batch)={loss.item():.6f}")


if __name__ == "__main__":
    main()
