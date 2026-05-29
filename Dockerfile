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

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.lock.txt
RUN pip install -r requirements.torcheeg.txt --no-deps

COPY . .

CMD ["python", "-m", "src.main"]
