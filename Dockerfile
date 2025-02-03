FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runpod and other direct dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application
COPY leeky ./leeky
COPY runpod_handler.py .

# Download spaCy model
RUN python3 -m spacy download en_core_web_sm

# Start the handler
CMD ["python3", "-u", "runpod_handler.py"]