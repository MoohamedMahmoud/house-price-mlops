FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY models ./models

RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

WORKDIR /app

RUN useradd \
    --create-home \
    --shell /usr/sbin/nologin \
    appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/models ./models

ENV PATH="/app/.venv/bin:$PATH"
ENV MODEL_PATH="/app/models/lasso_model.joblib"

EXPOSE 8000

USER appuser

CMD [ "uvicorn", "house_price_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
