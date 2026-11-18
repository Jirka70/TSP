FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt requirements.lock.txt
COPY requirements.torcheeg.txt requirements.torcheeg.txt
COPY requirements.cpu.txt requirements.cpu.txt
COPY requirements.tools.txt requirements.tools.txt

RUN python -m pip install -r requirements.tools.txt
RUN python -m pip install --build-constraint requirements.tools.txt -r requirements.lock.txt
RUN python -m pip check
RUN python -m pip install --build-constraint requirements.tools.txt -r requirements.torcheeg.txt --no-deps
RUN python -c "from braindecode.models import EEGNet"

COPY . .

CMD ["python", "-m", "src.main"]
