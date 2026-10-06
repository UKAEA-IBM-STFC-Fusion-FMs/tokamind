# GPO β Calibration Table (v8)
## Per-task β values, loss types, and shot blacklists

> **Source:** `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml` lines 17–691;
> `docs/gpo_implementation_log.md §3 (v8 table, lines 211–226)`;
> `docs/gpo_implementation_log.md §11 (per-task filters, lines 495–553)`

---

## Calibration Methodology

**β calibration criterion (v8):**
```
β_opt = 1 / p50_MSE
```
where `p50_MSE` is the median embedding-space MSE gap across all pairs for a signal,
measured from the collected `.npz` shards.

For multi-signal tasks, the **geometric mean** of per-signal `β_opt` is used.

**IPO target margin:** `1/(2β)` should be close to `p50_MSE`.  This ensures the IPO loss
has a reachable, well-calibrated target.

**Loss type selection:**
- `ipo` is used for nearly all tasks.  Rationale: DPO's sigmoid is flat (`≈ 0.5`) at
  initialisation when `β × margin ≈ 0`, providing no pair discrimination.  IPO's linear
  gradient `2·(margin − 1/(2β))` works at any scale.
- `dpo` is used for task_4-4 only, where `β × p50 ≈ 0.88` — well inside the DPO
  operating range from the start.

*Source: `gpo_tasks.yaml` comments lines 43–51; `gpo_implementation_log.md` lines 196–226*

---

## Per-Task β Table

| Task | Signal(s) | p50 MSE | β | β×p50 | IPO target `1/(2β)` | Loss | Notes |
|------|-----------|---------|---|-------|---------------------|------|-------|
| 4-1 | SXR lower + upper | 0.0789 | 10 | 0.79 | 0.050 | **IPO** | DPO→IPO in v7: reward hacking confirmed |
| 4-2 | SXR lower + upper | 0.0786 | 10 | 0.79 | 0.050 | **IPO** | Same signal family as 4-1 |
| 4-3 | EQ magnetic_axis_z | 0.0028 | 50 | 0.14 | 0.010 | **IPO** | Weak margin; β×p50 ≪ 0.5 at DPO init |
| 4-4 | summary-ip (plasma current) | 0.1095 | 8 | 0.88 | 0.063 | **DPO** | β×p50 in calibrated DPO range |
| 4-5 | magnetics omv+cc | 0.0223 | 5 | 0.11 | 0.100 | **IPO** | Fat tail; clip+log applied |
| 1-1 | EQ scalars (15 signals) | 0.003–0.010 | 100 | ≈0.5 | 0.005 | **IPO** | D=1 identity; geomean β_opt≈170 |
| 1-2 | LCFS (lcfs_r, lcfs_z) | 0.0132 / 0.0067 | 50 | 0.50 | 0.010 | **IPO** | Geomean p50≈0.0094 |
| 1-3 | EQ-ψ reconstruction | 0.2608 | 2 | 0.52 | 0.250 | **IPO** | IPO target 0.25 ≈ p50 |
| 2-1 | Multi-EQ (multiple outputs) | 0.0182 | 25 | 0.46 | 0.020 | **IPO** | Multi-output; geomean p50 |
| 2-2 | LCFS + NBI (lcfs_r, lcfs_z) | 0.0156–0.0367 | 40 | 0.50 | 0.013 | **IPO** | Geomean β_opt≈40 |
| 2-3 | EQ-ψ + actuators | 0.3055 | 2 | 0.61 | 0.250 | **IPO** | β=50 was oversaturated (σ=0.955) in v7; corrected to β=2 |

*Source: `gpo_tasks.yaml` lines 83–118 (4-1), 127–173 (4-2), 182–227 (4-3), 237–284 (4-4),
297–347 (4-5), 389–425 (1-1), 433–469 (1-2), 477–513 (1-3), 520–556 (2-1), 586–622 (2-2),
654–691 (2-3); also `gpo_implementation_log.md` lines 211–226*

---

## Per-Signal β Detail — task_1-1 (15 identity-encoded scalars)

Source: `gpo_tasks.yaml` lines 356–382

| Signal | p50 MSE | β_opt (=1/p50) |
|--------|---------|----------------|
| equilibrium-beta_normal | 0.0068 | 147 |
| equilibrium-beta_pol | 0.0053 | 189 |
| equilibrium-beta_tor | 0.0066 | 152 |
| equilibrium-bphi_rmag | 0.0094 | 106 |
| equilibrium-bvac_rmag | 0.0100 | 100 |
| equilibrium-elongation | 0.0077 | 130 |
| equilibrium-elongation_axis | 0.0091 | 110 |
| equilibrium-magnetic_axis_r | 0.0048 | 208 |
| equilibrium-magnetic_axis_z | 0.0012 | 833 |
| equilibrium-minor_radius | 0.0073 | 137 |
| equilibrium-q95 | 0.0033 | 303 |
| equilibrium-triangularity_lower | 0.0065 | 154 |
| equilibrium-triangularity_upper | 0.0087 | 115 |
| equilibrium-x_point_r | 0.0048 | 208 |
| equilibrium-x_point_z | 0.0001 | 7153 ← near-perfect; outlier excluded from geomean |

Geometric mean of β_opt (excluding x_point_z) ≈ **170**.  β=100 chosen: `1/(2×100)=0.005 ≈ p50` for most signals.

---

## Per-Signal β Detail — task_2-2 (LCFS + NBI)

Source: `gpo_tasks.yaml` lines 564–582

| Signal | p50 MSE | β_opt |
|--------|---------|-------|
| equilibrium-lcfs_r | 0.0367 | 25.2 |
| equilibrium-lcfs_z | 0.0156 | 64.1 |

Geometric mean: √(25.2 × 64.1) ≈ **40**.  IPO target: β=40 → `1/(2×40)=0.0125`:
- lcfs_r: target=0.0125 vs p50=0.037 — reachable (34% of p50)
- lcfs_z: target=0.0125 vs p50=0.016 — ideal (78% of p50)

---

## Shot Blacklists

Source: `gpo_tasks.yaml` lines 28–36, §§ per-task comments; `gpo_implementation_log.md §11`

| Task | Blacklisted shots | Reason |
|------|-------------------|--------|
| 4-1, 4-2 | 18675, 14013, 14342 | Mean MSE 45–54; p95 up to 188×; 1–3 windows each — unusual/disrupted discharges |
| 4-3 | *(none)* | Native NRMSE filter [25, 95] replaces historical `min_margin_mse: 0.01` |
| 4-4 | 18492, 14357, 17044, 12667, 15359, 16180, 15767, 15524, 14433, 12898, 20575, 20839, 14383, 17048, 17412 | All top-15 had n_windows=1 (single-window disruption captures); mean MSE 42–57 |
| 4-5 | 30339, 28311, 28268 | Mean MSE 117–284; p95 up to 1013 |
| 1-1 | 24463, 25327, 21296 | Mean MSE 41/36/10 — 367/324/86× dataset mean (0.1125) |
| 1-2 | 26921, 24828, 21359 | Top-3 from outlier table |
| 1-3 | 16914, 11822, 21004, 26846, 13202 | All have mean MSE > 100× dataset mean (background p50=0.26) |
| 2-1 | 24463, 22046, 23931 | Top-3; geometric-mean p50 MSE≈0.018 |
| 2-2 | 24828, 24463, 26921 | Mean MSE 6.1/2.6/2.3 — 70/30/27× dataset mean |
| 2-3 | 11822, 26259, 28240, 21004, 25094 | Mean MSE 1050/680/557/544/452 — 292/189/155/151/126× mean; Gini=0.901 |

**task_2-3 additional note:** Gini coefficient of 0.901 indicates extreme pair concentration.
Between-shot variance is only 13.2% of total — remaining gradient concentration is
within-shot (temporal).  Blacklisting beyond rank 5 has little structural benefit.

*Source: `gpo_implementation_log.md §11` lines 495–553*
