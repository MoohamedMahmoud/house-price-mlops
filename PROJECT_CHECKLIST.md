# MLOps Project Checklist — house-price-mlops (reusable template)

Status marked per item. Use this as the template for any future project.

## Phase 1 — Environment & Structure
- [x] `uv` project init, Python pinned
- [x] Folder structure: `src/`, `tests/`, `notebooks/`, `configs/`, `data/`, `models/`
- [x] `.gitignore` (data/, .venv/, secrets)
- [x] Dependencies declared + `uv.lock` committed
- [ ] `README.md` (still empty)
- [ ] Delete leftover `src/house_price_mlops/` scaffold folder

## Phase 2 — Data Understanding
- [x] EDA (distributions, correlations, categorical relationships)
- [x] Missing-value audit + documented exceptions
- [x] Outlier investigation (evidence-based, not visual-only)
- [x] Skew analysis → transform decisions

## Phase 3 — Preprocessing / Feature Engineering (notebook)
- [x] Missing-value handling decided
- [x] log1p transform columns decided
- [x] Encoding scheme decided (ordinal / binary / one-hot)
- [x] Outlier keep/drop decided (evidence: kept)
- [x] Notebook reviewed for bugs (2 found, fixed)

## Phase 4 — `src/` Extraction (notebook logic → tested code)
- [x] `preprocessing.py` (`AmesCleaner`) + tests
- [x] `features.py` (`AmesFeatureEngineer`) + tests
- [x] `pipeline.py` (`build_dataset` — outlier decision wired explicitly) + tests
- [x] `train.py` (baseline models + tuning configs) + tests
- [x] `evaluate.py` (metrics, diagnostics, CV) + tests
- [x] All src/ modules verified against real notebook outputs, not assumed

## Phase 5 — Model Selection
- [x] 10-model baseline comparison
- [x] Cross-validation (mean + std, not single split)
- [x] Hyperparameter tuning on top 3 candidates
- [ ] **Final model decision confirmed** (Lasso vs. Gradient Boosting — currently a near-tie)

## Phase 6 — Serving (not started)
- [ ] Persist trained model to `models/` (`joblib`)
- [ ] `src/api.py` — FastAPI `/predict` + `/health`, Pydantic input validation
- [ ] Structured logging (`structlog`)
- [ ] API-level tests (`TestClient`)

## Phase 7 — Containerization (not started)
- [ ] `Dockerfile` (multi-stage, non-root)
- [ ] `.dockerignore`
- [ ] `docker-compose.yml`

## Phase 8 — CI/CD (not started)
- [ ] `.github/workflows/ci.yml` — lint → test → build
- [ ] Coverage threshold gate

## Phase 9 — Polish (not started)
- [ ] `README.md` — setup, usage, architecture
- [ ] `configs/` populated (thresholds, paths — no magic numbers)

---

**You are here:** end of Phase 5. Next unblocking action: confirm final model → Phase 6.
