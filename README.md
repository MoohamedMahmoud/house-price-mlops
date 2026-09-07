<div align="center">

<br/>

# House Price Prediction — MLOps Service

**An end-to-end machine learning service that predicts Ames house sale prices, from notebook-driven data reasoning to a tested, containerized FastAPI API.**

Reproducible training pipeline · Semantics-aware preprocessing · Lasso regression · Joblib production artifact · ONNX validation · FastAPI · Structured logging · Docker Compose

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-validated-005CED?logo=onnx&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![uv](https://img.shields.io/badge/Package%20Manager-uv-DE5FE9)
![Tests](https://img.shields.io/badge/Tests-57%20passed-success)
![Coverage](https://img.shields.io/badge/Coverage-75.77%25-success)

<br/>

[Overview](#overview) · [Architecture](#architecture) · [Pipeline Stages](#pipeline-stages) · [Data Decisions](#data-decisions) · [Model Selection](#model-selection) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [Testing](#testing) · [Configuration](#configuration) · [Serialization](#serialization) · [Observability](#observability) · [Docker](#docker) · [Git & Release](#git--release) · [Project Structure](#project-structure) · [Tech Stack](#tech-stack) · [Contributing](#contributing)

<br/>

</div>

---

## Overview

This project productionizes a solution to the Kaggle **House Prices — Advanced Regression Techniques** competition using the Ames, Iowa housing dataset.

The project deliberately keeps the full reasoning trail: data semantics and EDA decisions are treated as the source of truth first, then extracted into reusable Python modules, tested, packaged, exposed through FastAPI, and containerized with Docker.

### Project goals

| Capability | Details |
|---|---|
| **Data reasoning** | Missing values are interpreted using Ames feature semantics instead of blanket imputation. |
| **Reproducible preprocessing** | `AmesCleaner` learns training-set statistics and reuses them during inference. |
| **Feature engineering** | Log transforms, ordinal mappings, binary encoding, and training-fitted one-hot vocabulary. |
| **Model selection** | Ten regression approaches compared on the competition's log-RMSE metric; final model is tuned Lasso. |
| **Production artifact** | `models/lasso_model.joblib` stores the fitted model plus preprocessing/feature metadata. |
| **API** | FastAPI exposes `/health`, `/metadata`, `/predict`, and `/predict/batch`. |
| **Observability** | Structured JSON logs and `X-Correlation-ID` request tracing. |
| **Serialization validation** | Joblib and ONNX prediction parity plus measured inference latency. |
| **Containerization** | Multi-stage Docker image, non-root runtime, Docker Compose healthcheck. |
| **Quality gate** | 57 pytest tests with an enforced minimum coverage of 70%. |

> **Who is this for?** Anyone learning how to take a real notebook-based regression project through the engineering steps required for a reproducible, testable, containerized MLOps service — without hiding the data decisions that made the model possible.

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                  House Price Prediction — MLOps Architecture                 │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────┐      ┌──────────────────────────┐
  │     DATA LAYER     │      │   OFFLINE TRAINING       │
  │                    │      │                          │
  │ data/raw/train.csv ├─────►│ load + validate          │
  │                    │      │ AmesCleaner              │
  │ data_description   │      │ AmesFeatureEngineer      │
  │ notebook decisions │      │ outlier decision         │
  └────────────────────┘      │ model selection/tuning   │
                              │ evaluation               │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │     MODEL ARTIFACTS      │
                              │                          │
                              │ lasso_model.joblib      │
                              │ lasso_model.onnx        │
                              └────────────┬─────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │         BACKEND — FastAPI :8000            │
                    │                                            │
                    │ Startup → load Joblib artifact             │
                    │                                            │
                    │ GET  /health                               │
                    │ GET  /metadata                             │
                    │ POST /predict                              │
                    │ POST /predict/batch                        │
                    │                                            │
                    │ Structured logs + correlation IDs           │
                    └──────────────────────┬─────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │   Docker / Docker Hub    │
                              │                          │
                              │ multi-stage image        │
                              │ non-root runtime         │
                              │ healthcheck               │
                              └──────────────────────────┘
```

### Serving architecture

The production API currently serves from **Joblib**, not ONNX:

```text
Raw Ames request
      ↓
Pydantic validation
      ↓
Missing-token normalization
      ↓
AmesCleaner
      ↓
AmesFeatureEngineer
      ↓
Exact 227 learned feature columns
      ↓
StandardScaler + Lasso from lasso_model.joblib
      ↓
log-price prediction
      ↓
expm1()
      ↓
SalePrice
```

ONNX is retained as a validated alternative serialization. Its prediction parity and isolated model-inference latency were measured, but the API was intentionally left on Joblib because the custom Ames cleaning/feature-engineering logic remains Python/pandas code either way.

---

## Pipeline Stages

### 1 — Data Understanding

The raw training dataset contains **1,460 rows × 81 columns**. `Id` is a unique identifier rather than a house characteristic, and duplicate rows were verified as zero.

The Kaggle metric is RMSE between log-transformed predicted and observed prices, so the project models:

```python
log1p(SalePrice)
```

and converts predictions back to the original dollar scale with:

```python
expm1(prediction_log)
```

### 2 — Data Cleaning

`AmesCleaner` implements the semantics-aware missing-value rules discovered during EDA. The key principle is that an Ames `NA` often means **feature absent**, not unknown.

Examples:

- `Alley = NA` → no alley access.
- basement `NA` → no basement for the relevant feature.
- garage `NA` → no garage.
- `FireplaceQu = NA` → no fireplace.
- `PoolQC = NA` → no pool.
- `Fence = NA` → no fence.

Real exceptions are handled separately before structural missing categories are filled.

### 3 — Feature Engineering

`AmesFeatureEngineer` applies three category strategies:

1. **Ordinal mappings** for ordered variables such as `ExterQual`, `KitchenQual`, and `BsmtQual`.
2. **Binary mapping** for `CentralAir` (`Y`/`N` → `1`/`0`).
3. **One-hot encoding** for nominal variables using vocabulary learned from the training data.

Nine genuinely skewed numeric columns receive `log1p`:

```text
GrLivArea
LotArea
MasVnrArea
1stFlrSF
2ndFlrSF
LotFrontage
BsmtFinSF1
WoodDeckSF
OpenPorchSF
```

### 4 — Outlier Decision

The raw-data anomaly rule is:

```python
(GrLivArea > 4000) & (SalePrice < 300000)
```

Two genuine anomalies were identified. A controlled keep-vs-remove experiment slightly favored **keeping** them, so the final training path explicitly uses:

```text
keep_outliers=True
```

### 5 — Model Selection

Ten regression approaches were compared on the same seeded 80/20 split and validated with 5-fold cross-validation.

The final model is:

```text
StandardScaler → Lasso
alpha = 0.003
max_iter = 10000
```

### 6 — Model Artifact

Training produces:

```text
models/lasso_model.joblib
```

The artifact contains:

- fitted model
- fitted `AmesCleaner`
- fitted `AmesFeatureEngineer`
- learned feature columns
- required raw input columns
- metadata

The final training run uses **1,460 rows and 227 processed features**.

### 7 — API Serving

FastAPI loads the artifact at startup. Requests are normalized and passed through the exact same learned preprocessing used during training.

### 8 — Serialization Validation

A separate ONNX artifact is generated and compared against Joblib for numerical parity and model-only inference latency.

### 9 — Containerization

The API is packaged into a multi-stage Docker image with a non-root runtime user and a `/health`-based container healthcheck.

---

## Data Decisions

This section summarizes the important dataset reasoning that should not be lost when the notebook is replaced by reusable production code.

### Missing values: structural absence vs unknown

A blanket `fillna()` is unsafe for Ames because many missing categorical values are intentional.

The project therefore treats missingness feature-by-feature.

### Real categorical exceptions

| Case | Rows | Decision |
|---|---:|---|
| `MasVnrType` missing while `MasVnrArea > 0` | 5: 624, 773, 1230, 1300, 1334 | Fill `MasVnrType` with training-set mode of known non-`None` values. |
| Partial basement gaps | 2: row 332 and row 948 | Fill the single missing field with its training-set mode rather than falsely declaring no basement. |

After those exceptions, remaining structural categorical missing values are represented as `None`/`"None"` according to the reusable cleaner logic.

### Numeric imputation rules

| Column | Rule | Why |
|---|---|---|
| `Electrical` | Training-set mode | Only one value is missing; categorical-like numeric-coded field is best handled from observed values. |
| `MasVnrArea` | `0` | Zero has the correct domain meaning: no veneer area. |
| `GarageYrBlt` | `YearBuilt` | Avoids inventing impossible year `0`; no garage has no garage construction year. |
| `LotFrontage` | Median by `Neighborhood`, with global fallback | Frontage is location-dependent, so group-wise statistics preserve local structure. |

### Category consistency

The actual dataset vocabulary is treated as the source of truth for encoding, not the prose description file alone.

Observed examples include:

- `BldgType`: `2fmCon` vs documentation wording such as `2FmCon`.
- `MSZoning`: raw `C (all)` vs documentation wording `C`.
- `Exterior1st` / `Exterior2nd`: differences such as `BrkComm` vs `Brk Cmn`, `CemntBd` vs `CmentBd`, and `WdShing` vs `Wd Shng`.

### Identifier handling

`Id` is removed before modeling because all 1,460 rows have unique IDs and the value itself has no house-level predictive meaning.

### Duplicate validation

Duplicate rows were checked and the raw training dataset contains **0 duplicates**.

---

## Model Selection

### Single-split comparison

The four strongest models on the validation log-RMSE comparison were linear/regularized approaches:

| Model | Train log-RMSE | Validation log-RMSE | Train/Val Gap |
|---|---:|---:|---:|
| **Lasso** | 0.09363 | **0.12078** | **0.02716** |
| Ridge | 0.09320 | 0.12132 | 0.02812 |
| Elastic Net | 0.09327 | 0.12116 | 0.02790 |
| Linear Regression | 0.09304 | 0.12259 | 0.02955 |

Tree models and SVR were worse on this particular feature representation, with several models showing severe train/validation gaps.

### 5-fold cross-validation

After tuning the strongest candidates on an apples-to-apples full-data CV comparison:

```text
Lasso                  0.12926
Gradient Boosting      0.12983
Hist Gradient Boosting ~0.1315
```

The margin is narrow, so the final choice is not presented as a dramatic accuracy win. Lasso was selected because it remained competitive while being simpler and easier to interpret and operate.

### Important modeling caveats

The baseline deliberately leaves several possible improvements deferred:

- binary `Has*` indicators for zero-inflated variables
- rare-neighborhood grouping
- cyclical encoding for `MoSold`
- broader hyperparameter tuning across every candidate model

These are future experiments, not silently applied changes.

---

## Quick Start

### Prerequisites

```text
Python 3.12+
git
uv
docker
Docker Compose
```

### Fastest published-container path

```bash
# 1. Pull the published image
docker pull m0hamedseleem/house-price-mlops:latest

# 2. Run the API
docker run --rm -p 8000:8000 m0hamedseleem/house-price-mlops:latest

# 3. Verify it is alive
curl http://localhost:8000/health
```

Expected health response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "Lasso"
}
```

### Local development

```bash
git clone https://github.com/MoohamedMahmoud/house-price-mlops.git
cd house-price-mlops
uv sync
```

Train/regenerate the artifact:

```bash
uv run python -m house_price_mlops.train
```

Run quality checks:

```bash
uv run pre-commit run --all-files
uv run pytest tests/ --cov=house_price_mlops --cov-fail-under=70
```

Run the local container stack:

```bash
docker compose build --no-cache
docker compose up -d
```

Stop it with:

```bash
docker compose down
```

---

## API Reference

Interactive FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

OpenAPI schema:

```text
http://localhost:8000/openapi.json
```

### `GET /health` — Health check

Checks API/model readiness.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "Lasso"
}
```

### `GET /metadata` — Model metadata

Returns the serving artifact and feature contract.

Example response shape:

```json
{
  "model_loaded": true,
  "model_type": "Pipeline",
  "feature_count": 227,
  "required_input_count": 79,
  "feature_columns": ["MSSubClass", "LotFrontage", "LotArea"],
  "required_input_columns": ["MSSubClass", "MSZoning", "LotFrontage"],
  "model_path": "/app/models/lasso_model.joblib"
}
```

### `POST /predict` — Predict `SalePrice`

Accepts the 79 raw Ames input features. The API ignores the `Id` identifier at the modeling layer.

Request:

```json
{
  "features": {
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotFrontage": 65,
    "LotArea": 8450,
    "Street": "Pave",
    "Alley": "NA",
    "Neighborhood": "CollgCr",
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "GrLivArea": 1710,
    "GarageCars": 2,
    "GarageArea": 548,
    "SaleType": "WD",
    "SaleCondition": "Normal"
  }
}
```

The complete 79-field contract is exposed by `/metadata` and `/docs`.

Response:

```json
{
  "prediction": 205780.79800261185,
  "prediction_log": 12.234571653390958
}
```

### `POST /predict/batch` — Batch prediction

Request shape:

```json
{
  "instances": [
    {"features": {"...": "..."}},
    {"features": {"...": "..."}}
  ]
}
```

Response:

```json
{
  "predictions": [
    {
      "prediction": 205780.79800261185,
      "prediction_log": 12.234571653390958
    }
  ]
}
```

### Missing-value input behavior

At the API boundary, these values are normalized before preprocessing:

```text
null
omitted field
""
"NA"
"N/A"
"NULL"
```

This prevents a literal string such as `"NA"` from accidentally becoming an actual category value during feature engineering.

### Correlation IDs

Every request accepts or receives:

```text
X-Correlation-ID
```

If the caller does not provide one, the API generates a UUID and returns it in the response headers.

---

## Testing

The test suite covers preprocessing, features, dataset assembly, evaluation, training, serialization, and FastAPI behavior.

Current verified result:

```text
57 passed
75.77% total coverage
coverage gate: >= 70%
```

Run the full gate:

```powershell
uv run pytest tests/ --cov=house_price_mlops --cov-fail-under=70
```

### Test design

The API tests include:

- health behavior when a model artifact is absent
- normal prediction
- `NA`, `N/A`, and empty-string normalization
- explicit JSON `null`
- omitted feature handling
- preservation of real values
- metadata contract
- batch prediction
- provided correlation IDs
- generated correlation IDs

Testing uses fixtures/fakes/mocks and parametrization where they make the behavior explicit rather than coupling tests to a live production artifact.

### Pre-commit

Run all configured hooks:

```powershell
uv run pre-commit run --all-files
```

Current hooks:

```text
ruff-check --fix
ruff-format
end-of-file-fixer
```

Black was intentionally removed from the hook chain so Ruff Formatter is the single Python formatting authority.

---

## Configuration

The API uses an environment variable for the model artifact path:

```text
MODEL_PATH=/app/models/lasso_model.joblib
```

Docker Compose supplies the container path, while local runs can override it as needed.

The project also uses `pyproject.toml` and `uv.lock` for reproducible dependency management.

---

## Serialization

### Joblib / Pickle

Production serving uses:

```text
models/lasso_model.joblib
```

It contains the complete serving state required by the current architecture:

```text
model
cleaner
engineer
feature_columns
required_input_columns
metadata
```

### ONNX

The fitted `StandardScaler → Lasso` pipeline is exported with `skl2onnx` to:

```text
models/lasso_model.onnx
```

The two formats were tested for numerical parity over 227 features.

| Format | Prediction | Mean latency |
|---|---:|---:|
| Joblib / sklearn | 1.431635521505 | 1.533652 ms (run 2) |
| ONNX Runtime | 1.431635856628 | 0.012461 ms (run 2) |
| Absolute difference | **3.351232e-07** | — |

Across two measurements, ONNX showed approximately **110–125×** faster isolated model inference on the development machine.

### Why the API still uses Joblib

The benchmark measures the model-inference operation itself, not the complete HTTP request. The API still performs JSON parsing, validation, custom `AmesCleaner` logic, and pandas-based feature engineering before model inference.

The current decision is therefore:

```text
Joblib → production serving ✅
ONNX   → validated alternative ✅
```

ONNX can be reconsidered later for a deployment target that specifically requires a portable non-scikit-learn runtime or for a demonstrated end-to-end performance requirement.

---

## Observability

The API emits structured JSON logs rather than application `print()` calls inside the API layer.

A completed request log includes fields such as:

```json
{
  "timestamp": "...",
  "level": "INFO",
  "logger": "house_price_mlops.api",
  "message": "Request completed: ...",
  "correlation_id": "...",
  "duration_ms": 1.23
}
```

Correlation IDs make it possible to connect request-start, request-completion, and client-side response information for the same request.

> Note: the broader repository still contains some `print()` calls in `train.py`; the API observability implementation itself uses structured logging. A full repository-wide no-`print()` cleanup remains a candidate follow-up.

---

## Docker

### Multi-stage image

The production `Dockerfile` uses two stages:

```text
builder
  ↓
install dependencies with uv
  ↓
create .venv
  ↓
copy application + model
  ↓
runtime
  ↓
non-root appuser
  ↓
uvicorn
```

The runtime stage copies only the built virtual environment, `src/`, and `models/`.

### Docker Compose

The Compose service is:

```text
house-price-api
```

It exposes:

```text
localhost:8000 → container:8000
```

The service includes a healthcheck against:

```text
/health
```

### Local container verification

The current image has been verified to:

- build successfully with `docker compose build --no-cache`
- start successfully
- load the Lasso artifact at startup
- report healthy
- return 200 from `/health`
- return 200 from `/metadata`
- return 200 from `/predict`
- return 200 from `/predict/batch`

Measured local image size:

```text
≈456 MB
```

### Docker Hub

Published image:

```text
m0hamedseleem/house-price-mlops:latest
```

The current published digest is:

```text
sha256:8214e9035d7247479236be5a014dee1cf9cee9b05f5e3604a51812cd89888a9a
```

The image was independently pulled and run from Docker Hub on host port `8001`, and `/health` and `/metadata` were verified successfully.

### Reproducible registry verification

```bash
docker pull m0hamedseleem/house-price-mlops:latest
docker run -d --name house-price-hub-test -p 8001:8000 m0hamedseleem/house-price-mlops:latest
curl http://localhost:8001/health
curl http://localhost:8001/metadata
```

After testing:

```bash
docker rm -f house-price-hub-test
```

---

## Git & Release

### Branch workflow

The project uses:

```text
main
  ↑
  └── feature branch → Pull Request → CI → Merge
```

The completed Module 1 work was merged into `main`, which is the repository's default branch.

### Useful inspection commands

```bash
git status
git branch -vv
git log --oneline --decorate -10
git remote -v
```

### Typical contribution flow

```bash
git switch -c feature/your-change
# make changes
uv run pre-commit run --all-files
uv run pytest tests/ --cov=house_price_mlops --cov-fail-under=70
git add <files>
git commit -m "feat: describe the change"
git push -u origin feature/your-change
# open a pull request
```

### Releases

The earlier Module 1 release remains tagged `v0.1.0`. The later completion work was merged into `main` as a newer commit; release versioning should therefore use a new version rather than reusing `v0.1.0`.

---

## Project Structure

```text
house-price-mlops/
├── data/
│   └── raw/
│       └── train.csv                  # local training dataset
├── notebooks/
│   └── 01_final_notebook_reviewed.ipynb
├── src/
│   └── house_price_mlops/
│       ├── __init__.py
│       ├── preprocessing.py            # AmesCleaner
│       ├── features.py                 # AmesFeatureEngineer
│       ├── pipeline.py                 # build_dataset()
│       ├── evaluate.py                 # evaluation + diagnostics
│       ├── train.py                    # training/model selection
│       └── api.py                      # FastAPI service
├── tests/
│   ├── test_api.py
│   ├── test_evaluate.py
│   ├── test_features.py
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_serialization.py
│   └── test_train.py
├── models/
│   ├── lasso_model.joblib
│   └── lasso_model.onnx
├── scripts/
│   ├── compare_serialization.py
│   └── export_onnx.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── uv.lock
├── README.md
└── module-1.md
```

The raw dataset itself is not embedded in this README; only its structure, decisions, and reproducibility contract are documented.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Dataset | Kaggle House Prices — Ames | Regression target: `SalePrice` |
| Python | Python 3.12 | Runtime |
| Package management | `uv` | Dependency resolution, locking, command execution |
| ML | scikit-learn | Scaling + Lasso regression |
| Data | pandas / NumPy | Cleaning, transformations, feature construction |
| Serialization | joblib | Production artifact |
| Model export | `skl2onnx` | ONNX conversion |
| Alternative runtime | ONNX Runtime | Validated inference alternative |
| API | FastAPI + Pydantic | Typed REST service |
| Testing | pytest + pytest-cov | Unit/API tests and coverage gate |
| Code quality | Ruff + pre-commit | Linting and formatting |
| Containers | Docker + Docker Compose | Reproducible runtime |
| Registry | Docker Hub | Published image |
| CI | GitHub Actions | Lint/test + Docker build |
| Version control | Git + GitHub | Branches, PRs, releases |

---

## Contributing

Pull requests are welcome.

For a change:

```bash
git switch -c feature/your-feature
# make changes
uv run pre-commit run --all-files
uv run pytest tests/ --cov=house_price_mlops --cov-fail-under=70
git add <files>
git commit -m "feat: add your feature"
git push -u origin feature/your-feature
```

Please keep the project principles intact:

- preserve train-serving consistency
- interpret Ames missingness semantically
- avoid changing preprocessing behavior without tests
- keep model feature columns deterministic
- verify every documented API endpoint after container changes
- do not claim unmeasured performance improvements

---

## Documentation & Evidence

The detailed engineering evidence is maintained in:

```text
module-1.md
```

It records the journey from notebook reasoning to reusable code, testing, API, serialization, Docker, Docker Hub, Git, and CI, including resolved bugs and remaining limitations.

The notebook and `data_description.txt` remain the references for dataset semantics and the original data-analysis decisions.

---

## License

MIT — see the repository license file.

---

<div align="center">

Built for the MLOps MENA Session 1 project · [Repository](https://github.com/MoohamedMahmoud/house-price-mlops)

</div>
