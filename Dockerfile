FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=1.4.2
ENV PATH="/opt/poetry/bin:$PATH"

# Configure poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION} \
    && poetry config virtualenvs.create false

# Set working directory
WORKDIR /app

# Copy project metadata files
COPY pyproject.toml poetry.lock README.md LICENSE ./

# Install dependencies without project
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy the project files
COPY leeky ./leeky
COPY runpod_handler.py ./

# Install the project itself
RUN poetry install --only main --no-interaction --no-ansi

# Download and cache spaCy model
RUN python -m spacy download en_core_web_sm && \
    python3 -c "import spacy; nlp = spacy.load('en_core_web_sm'); assert nlp, 'SpaCy model not loaded correctly'"

# Verify runpod is installed
RUN python3 -c "import runpod; print(f'RunPod version: {runpod.__version__}')"

# Start the handler
CMD ["python", "-u", "runpod_handler.py"]