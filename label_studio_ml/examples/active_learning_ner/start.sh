#!/bin/bash
# Start the Active Learning NER backend

# Create required directories
mkdir -p data/server/models data/.cache data/mlruns

# Check if running in Docker or directly
if [ "$1" == "docker" ]; then
    # Build and start Docker container
    echo "Starting with Docker..."
    docker compose up --build -d
    
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
    
    # Start the server
    echo "Starting server..."
    python _wsgi.py --port 9090 --host 0.0.0.0
fi 