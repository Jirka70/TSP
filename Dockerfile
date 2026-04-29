FROM python:3.11-slim

WORKDIR /app

COPY dist/*.whl /tmp/

RUN python -m pip install --upgrade pip \
    && python -m pip install /tmp/*.whl

CMD ["python", "-c", "import importlib.metadata as m; print(m.version('tsp-eeg-classification'))"]