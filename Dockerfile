FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies and clean up in the same layer
RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir runpod

# Set working directory
WORKDIR /app

# Copy only what's needed for dependency installation
COPY pyproject.toml poetry.lock README.md LICENSE ./

# Install Poetry and dependencies in one layer
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.4.2 \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root \
    && rm -rf ~/.cache/pypoetry

# Copy the project files
COPY leeky ./leeky
COPY runpod_handler.py ./

# Final installation and cleanup
RUN poetry install --only main --no-interaction --no-ansi \
    && python -m spacy download en_core_web_sm \
    && rm -rf ~/.cache/pip

# Start the handler
CMD ["python", "-u", "runpod_handler.py"]