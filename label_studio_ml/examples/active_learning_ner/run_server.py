#!/usr/bin/env python3
"""
Simple script to run the Label Studio ML backend server
"""

import os
import sys
from label_studio_ml.api import init_app

# Import the model
from label_studio_ml.examples.active_learning_ner.model import ActiveLearningNER

# Initialize the Flask app
def create_app():
    model_dir = os.environ.get('MODEL_DIR', '/data/models')
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    return init_app(
        model_class=ActiveLearningNER,
    )

# Create the app
app = create_app()

if __name__ == "__main__":
    # Get configuration from environment variables
    port = int(os.environ.get('PORT', 9090))
    host = os.environ.get('HOST', '0.0.0.0')
    workers = int(os.environ.get('WORKERS', 1))
    model_dir = os.environ.get('MODEL_DIR', '/data/models')
    
    print(f"Starting server on {host}:{port}")
    print(f"Model directory: {model_dir}")
    print(f"Workers: {workers}")
    
    # Use Flask's native run method for compatibility
    debug_mode = os.environ.get('DEBUG', '0') == '1'
    app.run(host=host, port=port, debug=debug_mode, threaded=True) 