FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11/uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY baitnswitch ./baitnswitch
COPY data ./data
# Train inside the image so the model always matches the installed scikit-learn.
RUN uv run --no-dev python -m baitnswitch.train

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "gunicorn", "-b", "0.0.0.0:8000", "-w", "2", "baitnswitch.app:create_app()"]
