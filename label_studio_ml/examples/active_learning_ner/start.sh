#!/bin/bash
# Start the Active Learning NER backend

# Create required directories
mkdir -p data/server/models data/.cache data/mlruns

# Check if .env file exists, create if not
if [ ! -f .env ]; then
    echo "Creating default .env file..."
    cat > .env << EOL
# Label Studio connection
LABEL_STUDIO_HOST=http://localhost:17777
LABEL_STUDIO_API_KEY=your_api_key_here

# Base model configuration
BASELINE_MODEL_NAME=dslim/bert-base-NER
FINETUNED_MODEL_NAME=finetuned_model

# Training configuration
START_TRAINING_EACH_N_UPDATES=10
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=3
WEIGHT_DECAY=0.01
MAX_SEQUENCE_LENGTH=128

# Active learning configuration
UNCERTAINTY_THRESHOLD=0.8
SAMPLES_PER_ITERATION=500

# MLflow configuration
MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# Basic authentication
BASIC_AUTH_USER=
BASIC_AUTH_PASS=

# Logging
LOG_LEVEL=INFO

# Server configuration
WORKERS=1
THREADS=8

# Model directory
MODEL_DIR=/data/models
EOL
    echo "Please edit the .env file to configure your settings."
    echo "Especially make sure to set your LABEL_STUDIO_API_KEY."
fi

# Check if running in Docker or directly
if [ "$1" == "docker" ]; then
    # Build and start Docker container
    echo "Starting with Docker..."
    docker compose up --build
    
    echo "Container is running at http://localhost:9090"
    echo "You can check logs with: docker logs active_learning_ner"
elif [ "$1" == "docker-run" ]; then
    # Run with Docker but without docker-compose
    echo "Starting with Docker run command..."
    
    # Load environment variables from .env
    export $(grep -v '^#' .env | xargs)
    
    docker run -d --name active_learning_ner \
        -p 9090:9090 \
        -v "$(pwd)/data/server:/data" \
        -v "$(pwd)/data/.cache:/root/.cache" \
        -v "$(pwd)/data/mlruns:/app/mlruns" \
        --env-file .env \
        heartexlabs/label-studio-ml-backend:active-learning-ner \
        bash -c "export PYTHONPATH=/app:\$PYTHONPATH && python /app/label_studio_ml/examples/active_learning_ner/run_server.py"
    
    echo "Container is running at http://localhost:9090"
    echo "You can check logs with: docker logs active_learning_ner"
else
    # Start directly with Python
    echo "Starting with Python..."
    
    # Check if requirements are installed
    if ! pip list | grep -q "label-studio-ml"; then
        echo "Installing requirements..."
        pip install -r requirements.txt
    fi
    
    # Load environment variables from .env
    export $(grep -v '^#' .env | xargs)
    
    # Start the server
    echo "Starting server..."
    
    # Make sure current directory is in Python path
    export PYTHONPATH=$(pwd):$PYTHONPATH
    
    # Run the server
    python -m label_studio_ml.examples.active_learning_ner._wsgi --port 9090 --host 0.0.0.0
fi 