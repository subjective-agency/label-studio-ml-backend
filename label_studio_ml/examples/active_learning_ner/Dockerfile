FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY label_studio_ml/examples/active_learning_ner/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files instead of installing the package
COPY label_studio_ml /app/label_studio_ml

# Add current directory to Python path
ENV PYTHONPATH=/app
ENV MODEL_DIR=/data/models
ENV PORT=9090

# Create necessary directories
RUN mkdir -p /data/models /data/cache

# Set default env vars
ENV BASELINE_MODEL_NAME=dslim/bert-base-NER
ENV FINETUNED_MODEL_NAME=finetuned_model
ENV START_TRAINING_EACH_N_UPDATES=1
ENV LEARNING_RATE=2e-5
ENV NUM_TRAIN_EPOCHS=3
ENV WEIGHT_DECAY=0.01
ENV MAX_SEQUENCE_LENGTH=128
ENV UNCERTAINTY_THRESHOLD=0.8
ENV SAMPLES_PER_ITERATION=500
# ENV MLFLOW_TRACKING_URI=sqlite:///mlruns.db
ENV WORKERS=1
ENV THREADS=8
ENV LOG_LEVEL=INFO

# Expose the port
EXPOSE 9090 