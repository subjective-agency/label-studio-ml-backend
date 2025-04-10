import os
import logging
import argparse
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.server import init_app

from model import ActiveLearningNER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_script_path = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.getenv('MODEL_DIR', os.path.join(_script_path, 'models'))
os.makedirs(_model_dir, exist_ok=True)

def get_kwargs_from_config():
    """Get model parameters from environment variables"""
    kwargs = {
        'batch_size': int(os.getenv('BATCH_SIZE', '16')),
        'learning_rate': float(os.getenv('LEARNING_RATE', '2e-5')),
        'max_sequence_length': int(os.getenv('MAX_SEQUENCE_LENGTH', '128')),
        'uncertainty_threshold': float(os.getenv('UNCERTAINTY_THRESHOLD', '0.8')),
        'samples_per_iteration': int(os.getenv('SAMPLES_PER_ITERATION', '500')),
    }
    return kwargs

def create_model_server():
    """Initialize model server"""
    return init_app(
        model_class=ActiveLearningNER,
        model_dir=_model_dir,
        model_kwargs=get_kwargs_from_config(),
        redis_queue=os.getenv('REDIS_QUEUE', 'default'),
        redis_host=os.getenv('REDIS_HOST', 'localhost'),
        redis_port=os.getenv('REDIS_PORT', '6379'),
        debug=bool(int(os.getenv('DEBUG', 0))),
    )

if __name__ == "__main__":
    """
    Start the model server using Docker or Python
    
    Example:
        python _wsgi.py --port 9090
    """
    parser = argparse.ArgumentParser(description='Label Studio ML Backend')
    parser.add_argument('--port', dest='port', type=int, default=9090, help='Server port')
    parser.add_argument('--host', dest='host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--workers', dest='workers', type=int, default=1, help='Server workers')
    parser.add_argument('--threads', dest='threads', type=int, default=8, help='Server threads')
    
    args = parser.parse_args()
    
    # Create model server
    app = create_model_server()
    
    # Start model server
    app.run(host=args.host, port=args.port, workers=args.workers, threads=args.threads) 