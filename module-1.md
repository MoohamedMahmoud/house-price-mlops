# Module 1 Report — House Price MLOps (Ames)

<!-- > Evidence document for Module 1. Numbers below are taken directly from
> `01_final_notebook_reviewed.ipynb`, the `src/house_price_mlops/` modules,
> and the measured results recorded in the project's own engineering-journey
> notes (`01-data-science-journey.md`, `02-mlops-engineering-journey.md`,
> `HOUSE_PRICE_MLOPS_PROJECT_CONTEXT.md`). Two items remain genuinely open —
> a real single-stage-vs-multi-stage Docker size comparison, and MAE — marked
> `[FILL IN]` below; everything else has been updated with real measured values. -->

---

## 1. Project Objective

Predict `SalePrice` for the Ames, Iowa housing dataset (Kaggle House Prices
competition), and productionize the winning model behind a FastAPI service,
containerized with Docker, with the model's own evaluation metric (log-RMSE)
as the basis for every modeling decision.

- **Dataset:** Kaggle House Prices — Ames, 1,460 rows × 81 columns (raw).
- **Target:** `SalePrice`, modeled as `log1p(SalePrice)` to match the
  competition's own evaluation metric.
- **Final model:** Lasso, `alpha = 0.003` (tuned; `alpha = 0.0005` in the
  original untuned baseline comparison).
- **Python / tooling:** Python 3.12, `uv` for environment/dependency management.
- **Package:** `src/house_price_mlops/`.

---

## 2. Repository Architecture

```
house-price-mlops/
├── src/house_price_mlops/
│   ├── preprocessing.py   # AmesCleaner — Stage 2/3: missing-value handling
│   ├── features.py        # AmesFeatureEngineer — Stage 5/6: log1p + encoding
│   ├── pipeline.py         # build_dataset() — assembles X/y, applies outlier decision
│   ├── train.py            # model definitions, tuning configs, final training
│   ├── evaluate.py         # evaluate_model(), plot_model_diagnostics(), CV helpers
│   └── api.py              # FastAPI service: /health, /metadata, /predict, /predict/batch
├── tests/                  # one test module per src module, pytest
├── notebooks/
│   └── 01_final_notebook_reviewed.ipynb   # EDA → decision, source of truth
├── models/                 # lasso_model.joblib (+ .onnx once exported)
├── Dockerfile               # multi-stage build (builder + slim runtime, non-root user)
├── docker-compose.yml
├── .github/workflows/ci.yml # lint, format check, pytest, docker build
└── pyproject.toml / uv.lock
```

---

## 3. Data Understanding & EDA Summary

- Raw shape confirmed: **1,460 rows × 81 columns**. `Id` is a pure row
  identifier (1,460 unique values) and is dropped explicitly, with `.shape`
  checked immediately after (81 → 80 columns) to verify the drop actually
  happened rather than assuming it.
- **0 duplicate rows.**
- `SalePrice` is strongly right-skewed: raw skew ≈ **1.88**; after `log1p`,
  skew drops to ≈ **0.12** (near-symmetric). This is the quantitative
  confirmation behind the decision to train on `log1p(SalePrice)` and invert
  with `expm1()` for any user-facing prediction.
- **Category-vs-documentation mismatches found** (raw values differ from
  `data_description.txt`'s stated vocabulary — encoders must always be fit on
  actual column values, never a hand-typed list):
  - `BldgType`: raw values `'2fmCon'`, `'Twnhs'` vs. documented `2FmCon`,
    `TwnhsI`.
  - `MSZoning`: raw `'C (all)'` vs. documented `C`.
  - `Exterior1st` / `Exterior2nd`: same material spelled two different ways
    across the two columns (`'WdShing'` vs `'Wd Shng'`, `'CemntBd'` vs
    `'CmentBd'`, `'BrkComm'` vs `'Brk Cmn'`), plus `Exterior2nd` has an extra
    `'Other'` category `Exterior1st` doesn't.
- **Correlation with `SalePrice`** (strongest linear signals):
  `OverallQual` 0.79, `GrLivArea` 0.71, `GarageCars` 0.64, `GarageArea` 0.62,
  `TotalBsmtSF` 0.61, `YearBuilt` 0.52.
- **Multicollinearity flagged** (correlated with each other, not just the
  target — relevant for linear models, irrelevant for tree models):
  `GarageCars`/`GarageArea` (0.88), `TotalBsmtSF`/`1stFlrSF` (0.82),
  `GrLivArea`/`TotRmsAbvGrd` (0.83).
- **Zero-inflated numeric columns** (raw magnitude gives a linear model
  almost nothing to learn from): `PoolArea` 99.5% zero, `3SsnPorch` 98.4%,
  `LowQualFinSF` 98.2%, `MiscVal` 96.4%, `BsmtHalfBath` 94.4%, `ScreenPorch`
  92.1%. Recommended (deferred, not yet implemented in `src/`) as binary
  `Has*` indicators rather than raw square footage.
- **Near-constant categorical columns:** `Utilities` is the extreme case —
  only **1 row out of 1,460** is not `'AllPub'`. `Street` (99.6%),
  `Condition2` (99.0%), `RoofMatl` (98.2%), and `Heating` (97.8%) are also
  effectively single-valued. `Utilities` is deliberately **not dropped** in
  `preprocessing.py` yet — this is an open Stage 3 decision, documented as
  such in the module docstring, not an oversight.
- `Neighborhood` has the widest category range (25 values) and uneven sample
  sizes across neighborhoods — flagged as a candidate for a rare-category
  "Other" bucket if a linear model is used (deferred, not implemented).

---

## 4. Missing Value Handling

**Principle:** most `NaN`s in this dataset are structural ("this feature
doesn't exist for this house"), not accidental — confirmed against
`data_description.txt`. A blanket `fillna()` would silently misencode the
real exceptions, so every group of "NaN means absent" columns was audited
row-by-row before applying the blanket rule.

**Two real exceptions found** (not just style nits — genuine data bugs the
blanket rule would have gotten wrong):

| Case | Rows affected | What was wrong with a blanket `'None'` fill | Fix |
|---|---|---|---|
| `MasVnrType` / `MasVnrArea` mismatch | 5 rows, index **624, 773, 1230, 1300, 1334** (`MasVnrArea` = 288, 1, 1, 344, 312) | These houses **do** have masonry veneer; the type just wasn't recorded. Filling `'None'` would falsely claim no veneer. | Fill with the mode of known non-`'None'` `MasVnrType` values, computed from training data only. |
| Basement partial-group gap | 2 rows (index 332: `BsmtFinType2` missing; index 948: `BsmtExposure` missing) | The rest of the basement fields are populated — these houses clearly have a basement — but exactly one field in the group is blank. Filling `'None'` would falsely claim no basement. | Fill the single missing field with the mode of known values for that field, learned from training data. |

**Raw missing-value counts, for reference** (before any fill — the counts
that motivated column-by-column treatment rather than one blanket rule):

| Column(s) | Missing count |
|---|---|
| `PoolQC` | 1,453 |
| `MiscFeature` | 1,406 |
| `Alley` | 1,369 |
| `Fence` | 1,179 |
| `MasVnrType` | 872 |
| `FireplaceQu` | 690 |
| `LotFrontage` | 259 |
| Garage fields (`GarageType`, `GarageYrBlt`, `GarageFinish`, `GarageQual`, `GarageCond`) | 81 each |
| Basement fields (`BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`) | 37–38 each |
| `MasVnrArea` | 8 |
| `Electrical` | 1 |

**Everything else** (14 remaining structural-`None` columns, `Electrical`
mode-fill, `MasVnrArea → 0`, `GarageYrBlt → YearBuilt`, `LotFrontage →`
per-neighborhood median with a global-median fallback) was verified correct
against the real data and applied as a documented, per-column rule — never
one global strategy. All of this logic lives in `AmesCleaner`
(`preprocessing.py`), fit on training data only and reused unchanged at
inference time, with `audit_structural_missingness()` available to re-run
the same systematic check on any new dataset before trusting a blanket fill.

Post-fill: **0 missing values** across all columns, verified with
`train.info()`.

---

## 5. Outlier Investigation

`GrLivArea` vs `SalePrice` shows 4 candidate outliers above 4,000 sq ft — but
only 2 are genuine anomalies:

| GrLivArea | SalePrice | Verdict |
|---|---|---|
| 4,316 sq ft | $755,000 | Legitimate — large, expensive house that fits the trend. **Kept.** |
| 4,476 sq ft | $745,000 | Legitimate — same as above. **Kept.** |
| 4,676 sq ft | $184,750 | Genuine anomaly (`OverallQual`=10 but priced far below trend). |
| 5,642 sq ft | $160,000 | Genuine anomaly (same profile). |

Filtering on `GrLivArea > 4000` alone would have wrongly dropped the two
legitimate expensive houses. The correct filter combines both conditions:

```python
mask = (raw_df["GrLivArea"] > 4000) & (raw_df["SalePrice"] < 300000)
```

**The two genuine outliers were not dropped automatically just because they
looked extreme.** The decision was settled by a controlled experiment: train
with the rows, train without them, evaluate both on an *identical* validation
set that excludes them either way (so the score difference is attributable
only to training-set composition). Result: evidence very slightly favored
**keeping** the two rows, so `keep_outliers=True` is the final decision,
now enforced by an explicit, required argument to `build_dataset()` (see
Bug Log, §16) rather than left to a default.

---

## 6. Feature Engineering & Encoding

Three column types, three different treatments — never one blanket encoder:

- **Ordinal** (`ExterQual`, `KitchenQual`, `BsmtQual`, etc.) — hand-written
  integer maps matching the documented order (`Po < Fa < TA < Gd < Ex`).
  Confirmed visually first: both `KitchenQual` and `ExterQual` show a clean,
  monotonic price step from `Po` to `Ex`.
- **Binary** — `CentralAir` (`Y`/`N`) mapped directly to `1`/`0`.
- **Nominal** (`Neighborhood`, `MSZoning`, etc.) — one-hot, vocabulary
  learned from training data only and reused unchanged on any other data
  (`AmesFeatureEngineer`), so train/test always produce identical columns
  even if a category's frequency differs or a category is entirely absent
  from one side.
- **`sklearn.LabelEncoder` was deliberately not used anywhere** — it assigns
  integers by sort order with no regard for whether that order is
  meaningful, which would scramble the real quality order on ordinal columns
  and impose a false distance between categories on nominal ones.

**`log1p` transform** applied to 9 continuous, genuinely skewed columns,
selected by a measured rule (not assumption): `|raw skew| > 0.75` **and**
the transform reduces that skew by at least 30%.

| Included (log1p applied) | Excluded (log1p made skew worse) |
|---|---|
| `GrLivArea`, `LotArea`, `MasVnrArea`, `1stFlrSF`, `2ndFlrSF`, `LotFrontage`, `BsmtFinSF1`, `WoodDeckSF`, `OpenPorchSF` | `TotalBsmtSF` (+1.52 → **-5.15**), `GarageArea`, `BsmtUnfSF` — all zero-inflated, so `log1p` compresses the nonzero tail while leaving the pile of zeros untouched, distorting rather than fixing the shape. |

**Years/dates left deliberately untransformed** (`YearBuilt`, `YearRemodAdd`,
`GarageYrBlt`, `YrSold`, `MoSold`) — a year isn't skewed in the sense `log1p`
addresses; it's a point on a timeline, not a magnitude with a long tail.
`MoSold`'s cyclical nature (December and January are adjacent, not 11 apart)
is named as a real, deliberately **deferred** refinement — not applied in
this baseline.

---

## 7. Model Selection & Evaluation

Ten regression approaches across three families (plain/regularized linear,
single tree + tree ensembles, one kernel method), all evaluated on an
identical, seeded 80/20 train/validation split (`random_state=42`).
**Regression-appropriate metrics only** — log-RMSE (competition's own
metric, and the only metric used to rank models) and RMSE in dollars
(`expm1()`'d back, for human interpretability only, never used to rank).
Scaling (`StandardScaler`) applied only to the five scale-sensitive models
(linear/regularized + SVR); left off tree-based models, which split on
per-feature thresholds and are scale-invariant.

### Single-split comparison (Stage 10/11)

| Model | Train log-RMSE | Validation log-RMSE | Train/Val Gap |
|---|---|---|---|
| **Lasso** | 0.09363 | **0.12078** | **0.02716** |
| Ridge | 0.09320 | 0.12132 | 0.02812 |
| Elastic Net | 0.09327 | 0.12116 | 0.02790 |
| Linear Regression | 0.09304 | 0.12259 | 0.02955 |
| Gradient Boosting | 0.07577 | 0.13764 | 0.06187 |
| Hist Gradient Boosting | 0.04162 | 0.13793 | 0.09631 |
| Extra Trees | 0.00000 | 0.14301 | 0.14301 |
| Random Forest | 0.05342 | 0.14495 | 0.09153 |
| Decision Tree | 0.00000 | 0.18385 | 0.18385 |
| SVR | 0.07817 | 0.21650 | 0.13833 |

**Reading it:** the four linear/regularized models occupy the top four
positions (0.1208–0.1226), all with small train/val gaps (0.027–0.030).
Every tree-based model and SVR scores meaningfully worse (0.138–0.217),
several with **exactly 0** training error — textbook overfitting (an
unconstrained Decision Tree/Extra Trees memorizes every training row). This
is a finding specific to this dataset and feature representation (a
high-dimensional, mostly one-hot, sparse matrix, with a target that behaves
fairly linearly on the log scale) — not a general claim that tree models are
worse than linear models.

### 5-fold cross-validation (Stage 11) — checking the single-split result held

- **Untuned:** Lasso still ranked #1 by mean, but the single-split "clear
  win" mostly disappeared — Lasso 0.1331 vs. Gradient Boosting 0.1342, a gap
  of only 0.0011, smaller than Lasso's own fold-to-fold std (0.0286).
- Lasso has one of the *largest* stds in the table (0.0286); Hist Gradient
  Boosting's is smaller (0.0146), Extra Trees' smallest of all (0.0107) —
  worth noting directly, since lower std is a real reliability signal, not
  a footnote to the mean.
- **Tuned, apples-to-apples** (both evaluated on the same full `X`/`y` and
  the same `KFold`): **Lasso 0.12926 vs. Gradient Boosting 0.12983** — Lasso
  still wins, but only narrowly, well inside the fold-to-fold noise
  (std ≈ 0.02–0.03). Hist Gradient Boosting ≈ 0.1315.

---

## 8. Final Model Decision

**Lasso, `alpha = 0.003`** (tuned via `GridSearchCV` over
`[0.0001 … 0.1]`), on lowest validation log-RMSE and smallest train/val gap,
confirmed by both the single split and 5-fold CV.

**Honest caveats, carried forward from the notebook rather than smoothed
over:**
- None of the ten models were tuned in the initial comparison — only the
  top 3 (Lasso, Gradient Boosting, Hist Gradient Boosting) were tuned
  afterward. SVR in particular likely suffered from running untuned.
- Several deferred feature-engineering ideas (`Has*` indicators, rare-
  neighborhood grouping, cyclical `MoSold`) were never applied — any of
  them could shift the ranking, especially for tree-based models.
- Lasso and tuned Gradient Boosting are close enough (0.12926 vs. 0.12983)
  that either would be a reasonable production candidate — the deciding
  factor is Lasso's simplicity/interpretability, not a meaningful accuracy
  gap.

**Note on metric terminology:** the handbook checklist asks for "before/after
MAE," but this project's notebook and `evaluate.py` consistently use
**log-RMSE** (matching the competition's own metric) as the ranking metric,
plus RMSE in dollars for interpretability — MAE was never computed anywhere
in the pipeline. `[FILL IN: decide whether to compute MAE retroactively for
the report, or note the metric substitution explicitly and cite log-RMSE as
the actual decision basis.]`

---

## 9. Production Code Extraction (Notebook → `src/`)

Extraction happened only after the notebook's own logic was treated as
validated — never the reverse:

| Notebook stage | `src/` module | Status |
|---|---|---|
| Stage 2/3 (missing values) | `preprocessing.py` → `AmesCleaner` | Done, tested |
| Stage 5/6 (log1p + encoding) | `features.py` → `AmesFeatureEngineer` | Done, tested |
| Stage 7/8 (outlier decision + assembly) | `pipeline.py` → `build_dataset()` | Done, tested |
| Stage 10.1 (evaluation) | `evaluate.py` | Done, tested |
| Stage 10/11 (model defs + training) | `train.py` | Done, tested |
| — | `api.py` (FastAPI service) | Done, tested |

Each extracted module has its own `pytest` suite proving it reproduces the
notebook's actual behavior — the same pattern the notebook itself names as
the bar for extraction (`assert_frame_equal` checks against `AmesCleaner`).

---

## 10. API & Serving

- `FastAPI` app with `/health`, `/metadata`, `/predict`, `/predict/batch`.
- Pydantic request validation (`extra="forbid"` on request models).
- Missing-value normalization at the API boundary — `null`, omitted field,
  `""`, `"NA"`, `"N/A"`, `"NULL"` are all normalized to `None` before
  reaching `AmesCleaner`, so a caller's literal string `"NA"` can't leak
  through as a category value.
- Model/preprocessing artifact loaded once at startup (`@app.on_event`).
- Structured JSON logging with a correlation ID
  (`X-Correlation-ID` request/response header, generated with `uuid4` if not
  supplied by the caller) around every request.

---

## 11. Serialization: joblib vs. ONNX

- `models/lasso_model.joblib` — the full artifact (model + fitted
  `AmesCleaner` + fitted `AmesFeatureEngineer` + feature columns + required
  input columns + metadata).
- `export_onnx.py` converts the fitted `Pipeline(StandardScaler, Lasso)` to
  ONNX via `skl2onnx`, `target_opset=17`.
- `test_serialization.py` asserts numerical parity between the two formats
  (`np.testing.assert_allclose`, `rtol=1e-5`).
- `compare_serialization.py` benchmarks both formats over 1,000 runs.

Measured on **227 features**, 1,000 runs per format:

| | Prediction | Latency (mean, 1000 runs) |
|---|---|---|
| Pickle / joblib | `1.431635521505` | `1.647304` ms |
| ONNX Runtime | `1.431635856628` | `0.014903` ms |
| Speedup | — | **110.53x** |

Absolute prediction difference: `3.351232e-07` — confirms numerical parity
well inside the test's tolerance (`test_serialization.py`,
`np.testing.assert_allclose(rtol=1e-5, atol=1e-6)`).

**One real bug fixed in this benchmark:** the first run passed a raw NumPy
array to the sklearn side of the comparison, but the pipeline was fit with
named columns (`StandardScaler` + `Lasso` trained on a `DataFrame`) —
this produced repeated sklearn feature-name warnings. Fixed by keeping a
`DataFrame` with the original feature names for the sklearn/pickle path,
and converting to `float32` NumPy only for the ONNX Runtime path, since
the two runtimes have different input-format expectations.

**Operational note:** `models/*.onnx` is `.gitignore`d by default (model
artifacts generally shouldn't live in git history); the ONNX file was
intentionally force-added for this project with
`git add -f models/lasso_model.onnx` so the parity test has something to
load in CI. Worth revisiting once model versioning/artifact storage is
handled by something other than git (e.g. an artifact registry).

---

## 12. Testing & Quality Gate

- `pytest` suite: fixtures, fakes/mocks (e.g. `FakeCleaner`, `FakeModel` in
  `test_api.py`), parametrization for missing-value cases.
- Unit tests cover preprocessing, features, pipeline, evaluation, training,
  and the API.
- **57 tests passing locally** (grew from 52 after the `/metadata`,
  `/predict/batch`, correlation-ID, and serialization-parity tests were
  added); CI Lint/Test job green.
- **Coverage gate is wired in and passing:**

  ```powershell
  uv run pytest tests/ --cov=house_price_mlops --cov-fail-under=70
  ```

  Measured result: **75.77% total coverage**, gate `>= 70%` **passed**.
- 33 non-failing warnings currently observed, worth a future cleanup pass
  rather than urgent: Starlette/httpx `TestClient` deprecation, FastAPI
  `@app.on_event("startup")` deprecation (migrate to `lifespan`), and a
  joblib/NumPy serialization warning.

---

## 13. Docker & Containerization

- Multi-stage `Dockerfile`: `python:3.12-slim` builder installs deps via
  `uv sync --frozen --no-dev`; runtime stage copies only the built venv,
  `src/`, and `models/`.
- Non-root `appuser` in the runtime stage.
- `MODEL_PATH` supplied via environment variable, not hardcoded.
- Healthcheck via `urllib.request` against `/health`.
- `docker compose build --no-cache` and `docker compose up` both verified;
  container's `/health`, `/predict`, `/docs`, and `/openapi.json` all return
  HTTP 200.
- CI runs a Docker build job on every push/PR.
- Local multi-stage image (`house-price-mlops-house-price-api:latest`)
  measures **≈456 MB**.
- Docker Desktop flagged a base-image vulnerability warning on
  `python:3.12-slim` — noted, not yet remediated (candidate future work:
  pin a patched slim digest or move to a distroless/hardened base).

| | Image size |
|---|---|
| Single-stage (comparison build, no multi-stage split) | `[FILL IN — not yet built; would need a throwaway single-stage Dockerfile for a real A/B comparison]` |
| Multi-stage (actual, current `Dockerfile`) | **≈456 MB** (measured) |
| Difference | `[FILL IN once the single-stage comparison image exists]` |

**Docker Hub image (published, verified pullable):**

```text
m0hamedseleem/house-price-mlops:latest
```

- Push succeeded; digest:
  `sha256:f42045adf74ebb0b9f9f0abe01775c5e5d39f281ccef968912797293e7eb9ee3`.
- **Independently verified**: pulled the image fresh with
  `docker pull m0hamedseleem/house-price-mlops:latest`, ran it on host port
  8001 (a different port than the local dev container), and `/health`
  returned successfully — this is stronger evidence than only confirming
  the locally-built image starts, since it proves the *published* artifact
  works independently of the local build environment.

---

## 14. CI/CD, Git & Release Workflow

- `main` branch, feature work on `module-1-packaging` → PR → merge.
- CI (`ci.yml`): `ruff format --check`, `ruff check`, `pytest`, then a
  separate Docker build job.
- Release tagged `v0.1.0` (annotated tag, pushed to GitHub).
- Git history includes real, resolved issues (see Bug Log below) rather
  than a rewritten-clean history — kept intentionally as evidence of actual
  debugging, per the project's own "evidence, not just code" rule.

---

## 15. Definition of Done — Current Status

Legend: ✅ done · 🟡 partial · ⬜ not started (pulled from
`README_Module1_Checklist.md`, Phase 14).

| Item | Status |
|---|---|
| Public GitHub repo, PR merged, CI green, `v0.1.0` tagged | ✅ |
| Docker image builds locally + in CI | ✅ |
| Local/container API works (`/health`, `/predict`) | ✅ |
| `/metadata`, `/predict/batch` | ✅ (implemented in `api.py`, per this review — recheck against original checklist date) |
| Structured JSON logging + correlation IDs | ✅ (implemented in `api.py`; verify against handbook's DEBUG/INFO/WARNING/ERROR level rules) |
| Zero `print()` in `src/` | 🟡 (`train.py` still uses `print()` in `train_final_model()`; recheck full `src/` before declaring done) |
| ONNX export + parity test + latency comparison | ✅ (parity: 3.35e-07 abs diff; latency: 110.53x speedup — §11) |
| 70% coverage gate in CI | ✅ (75.77% measured, gate passing) |
| pre-commit hooks installed and verified | ✅ (`ruff-check`, `ruff-format`, `end-of-file-fixer`; `pre-commit run --all-files` passes) |
| Docker image-size comparison | 🟡 (multi-stage size measured, ≈456MB; single-stage comparison build still not built — §13) |
| Docker Hub public image + external pull test | ✅ (`m0hamedseleem/house-price-mlops:latest`, pulled + run independently on port 8001 — §13) |
| 3-command README | ⬜ |
| `reports/module-1.md` | 🟡 (this document — MAE and single-stage image size are the two remaining genuine gaps) |
| Maturity self-assessment | see §18 below |
| Peer review, `module-1-completion` branch push + PR/merge | ⬜ (branch has local commits `6218eb5`, `1457059`, `1ecd504`; not yet pushed/merged) |

---

## 16. Bug & Failure Log

Kept as evidence per the project's own golden rule — a real log of what
broke and how it was diagnosed, not a cleaned-up success narrative.

**Notebook / EDA:**
- `MasVnrType`/`MasVnrArea` mismatch (5 rows) — a blanket structural-`None`
  fill would have wrongly encoded "no veneer" for houses that provably have
  one. Caught by cross-checking the categorical column against its paired
  numeric column, not by trusting the per-column missing count alone.
- Basement partial-group gap (2 rows, index 332 & 948) — same class of bug,
  found by checking whether missingness is *consistent within* a column
  group, not just plausible per-column.
- **Stage 7 outlier decision was computed but never wired into `final_df`**
  — the keep/drop experiment printed a decision, but the modeling dataframe
  was always built from the full encoded table regardless. Invisible in the
  original run only because the decision happened to be KEEP (a no-op).
  Fixed by making `keep_outliers` a required, explicit argument to
  `build_dataset()` — there is no way to call it without stating the
  decision.
- **Post-tuning CV comparison used a 1,168-row subset (`X_train`) while
  reusing a `KFold` object built for 1,460 rows** — not the same partition,
  not comparable to the untuned CV table it was meant to be compared
  against. As originally written this made Gradient Boosting look better
  than Lasso (0.1270 vs 0.1326); the corrected, apples-to-apples comparison
  shows Lasso still narrowly ahead (0.12926 vs 0.12983). Fixed by evaluating
  on the same full `X`/`y` used everywhere else in that stage.
- Two independent definitions of the same 10 models (one inline in the
  notebook, one imported from `src/train.py`) risked silent hyperparameter
  drift between the single-split and CV tables. Consolidated to one
  source of truth.

**Environment / packaging:**
- `uv add matplotlib.pyplot seaborn` failed — `matplotlib.pyplot` is a
  submodule, not an installable package; corrected to `matplotlib`.
- Project code reorganized from a flat `src/preprocessing.py` into the
  `house_price_mlops` package namespace.
- Serialized artifact originally pickled with module path `src.preprocessing`
  — broke on reload after the package rename. Fixed by retraining the
  artifact under the corrected `house_price_mlops.*` import path.

**API / testing:**
- Literal `"NA"` could reach feature engineering as a real category token
  instead of a missing value — fixed by normalizing recognized
  missing-value strings to `None` at the API boundary
  (`normalize_missing_value()`).
- An API test originally inspected the wrong `FakeCleaner` instance because
  the API loads a deserialized copy from the joblib artifact, not the
  original Python object passed to `joblib.dump`. Test was corrected to
  inspect the actually-loaded cleaner (`api._cleaner`).
- Exact floating-point equality on a predicted price was recognized as a
  fragile test pattern; replaced with `pytest.approx`.

**Docker / Git:**
- Docker Desktop's daemon wasn't running on first `docker compose build` —
  connection error, not a build error.
- Docker build initially failed because `uv sync` expected the package at a
  path that didn't match the actual repo layout — fixed by correcting the
  package structure.
- `origin` remote didn't exist initially — `git remote add origin ...`.
- Switching to the old `main` branch prompted a `notebooks/` deletion
  warning — answered `n`, returned to the working branch, synchronized
  after merge instead.
- Local `main` lacked upstream tracking — fixed with
  `git branch --set-upstream-to=origin/main main`.
- CI failed once on formatting (`ruff format --check`) on `tests/test_api.py`
  — diagnosed from GitHub Actions logs, fixed by running the formatter
  locally and committing the result.

**Pre-commit:**
- Initial `.pre-commit-config.yaml` was empty/invalid, raising
  `InvalidConfigError` before any hook could run.
- Ruff, once configured, scanned the notebook by default and reported
  `F821` (undefined name) errors from variables that only make sense in
  notebook execution order — not real bugs, just a scope mismatch. Fixed by
  scoping the Ruff hooks explicitly to `src`, `tests`, and `scripts`,
  leaving the notebook out of static-lint scope entirely.
- Ruff Formatter and Black were both configured initially and repeatedly
  reformatted the same file (`tests/test_serialization.py`) differently on
  each run — a formatter-vs-formatter fight rather than either being wrong.
  Resolved by removing Black and keeping Ruff Formatter as the single
  Python formatter, so there's exactly one source of truth for style.

**Serialization:**
- First ONNX benchmark run passed a raw NumPy array to the sklearn/pickle
  side, which had been fit on a named `DataFrame` — produced repeated
  sklearn feature-name warnings. Fixed by keeping the `DataFrame` (with
  feature names) for the sklearn path and only converting to `float32`
  NumPy for the ONNX Runtime path.

---

## 17. Known Limitations & Remaining Work

- Deferred feature engineering: `Has*` binary indicators for zero-inflated
  columns, rare-`Neighborhood` grouping, cyclical `MoSold` encoding — named
  and justified in the notebook, not yet implemented in `src/`.
- `Utilities` is retained despite carrying almost no signal (99.9% one
  value) — an open Stage 3 decision, not a bug.
- Model comparison used reasonable defaults for 7 of 10 models; only the
  top 3 were hyperparameter-tuned.
- `print()` statements remain in `train.py`; structured logging currently
  covers `api.py` only — audit the rest of `src/` before declaring "zero
  `print()`" complete.
- A genuine single-stage-vs-multi-stage Docker image-size comparison hasn't
  been produced yet — only the multi-stage size (≈456 MB) is measured; a
  throwaway single-stage `Dockerfile` would be needed for the actual A/B.
- MAE (as requested by the course handbook's PR template) was never
  computed — the project's actual decision metric throughout is log-RMSE
  (matching the competition's own scoring). Either compute MAE retroactively
  from the existing validation predictions, or document explicitly that
  log-RMSE was used as the substituted decision metric and why.
- The `module-1-completion` branch has commits (`6218eb5`, `1457059`,
  `1ecd504`) that are not yet pushed to GitHub, so none of this newer work
  (batch endpoint, logging, ONNX benchmark) has gone through PR/CI yet —
  only the earlier `v0.1.0` release has.
- Known non-failing deprecation warnings (Starlette/httpx `TestClient`,
  FastAPI `on_event`) and an unremediated base-image vulnerability warning
  on `python:3.12-slim` are documented but not yet acted on.

<!-- ---

## 18. Maturity Self-Assessment

`[FILL IN — this needs your own judgment, not something I should assert on
your behalf. Suggested axes to score, e.g. 1-5 each, with one line of
justification per axis: reproducibility, test coverage, observability/
logging, deployment automation, documentation completeness, data/model
governance (train-serving skew prevention, versioning).]` -->
