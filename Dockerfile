FROM python:3.10.13-slim-bullseye

ENV PYTHONUNBUFFERED 0
ENV PYTHONPATH /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry
RUN pip install streamlit

WORKDIR /app

COPY pyproject.toml poetry.lock mypy.ini ./
COPY evo_prophet ./evo_prophet

RUN poetry install

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["poetry", "run", "streamlit", "run", "evo_prophet/app.py", "--server.port=8501", "--server.address=0.0.0.0"]