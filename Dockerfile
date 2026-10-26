# UpHill: React UI + FastAPI backend in one image (Render web service).
FROM node:22-slim AS web
WORKDIR /web
COPY UpHill/web/package.json UpHill/web/package-lock.json ./
RUN npm ci
COPY UpHill/web/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src:/app/UpHill
WORKDIR /app
# Dependencies only: the app runs from source so UpHill/data and UpHill/web/dist resolve next to the package.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > requirements.txt \
    && pip install -r requirements.txt
COPY src/ src/
COPY UpHill/uphill/ UpHill/uphill/
COPY UpHill/data/ UpHill/data/
COPY --from=web /web/dist UpHill/web/dist
RUN useradd --create-home uphill && chown -R uphill /app/UpHill/data
USER uphill
ENV UPHILL_HOST=0.0.0.0 UPHILL_PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "python -m quantterm.cli migrate && exec python -m uphill"]
