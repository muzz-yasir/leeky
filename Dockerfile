FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install base dependencies first
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    torch==2.0.1 \
    -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Install pydantic v1 explicitly before other dependencies
RUN pip3 install --no-cache-dir "pydantic<2.0.0"

# Install spacy with specific version
RUN pip3 install --no-cache-dir \
    spacy==3.5.0 \
    && python3 -m spacy download en_core_web_sm

# Copy and install other requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application
COPY leeky ./leeky
COPY runpod_handler.py .

# Start the handler
CMD ["python3", "-u", "runpod_handler.py"]