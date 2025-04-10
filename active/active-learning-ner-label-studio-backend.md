# Active Learning NER with Label Studio ML Backend

## Overview

This document captures the planning and implementation details for creating a custom active learning Named Entity Recognition (NER) backend for Label Studio. The goal is to implement a system that can efficiently build a large NER dataset (approx. 30,000 records) starting with only ~200 manually labeled samples.

## Current Implementation

We currently have an `ActiveLearningPipeline` class in `components/wapaganda/active-learning-ner/core.py` that orchestrates:

1. Model training
2. Uncertainty estimation 
3. Sample selection
4. Label Studio integration
5. MLflow experiment tracking

The base components are located in `bases/wapaganda/active_learning/`:
- `settings.py` - Configuration using pydantic-settings
- `model_trainer.py` - NER model training with Hugging Face transformers
- `uncertainty_estimator.py` - Logic for uncertainty calculation
- `label_studio_connector.py` - Integration with Label Studio

We've also created a "selfish script" in `selfish/active_learning_ner.py` that provides a CLI interface for:
- Training the initial model (Phase 2, Step 4)
- Selecting uncertain samples (Phase 3, Step 5)
- Updating the model with new annotations (Phase 3, Step 7)

## Label Studio ML Backend Approach

After reviewing the [Label Studio ML Backend repository](https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/examples/huggingface_ner/README.md), we've decided to implement our active learning pipeline as a custom ML backend.

### Benefits of the ML Backend Approach

1. **Full integration with Label Studio** - Direct API communication
2. **Built-in serving infrastructure** - With API endpoints, authentication, and containerization
3. **Automatic prediction requests** - Label Studio requests predictions for new data
4. **Seamless experience for annotators** - Pre-annotations appear directly in the interface

### Implementation Plan

1. Create a new repository based on the Label Studio ML Backend template
2. Implement a custom `model.py` file that:
   - Uses our active learning logic for uncertainty estimation and sample selection
   - Integrates with Label Studio's API for predictions and training
   - Tracks uncertainties for active learning
   - Provides model training with proper evaluation metrics

### Core Backend Implementation

Here's the proposed structure for the `model.py` file:

```python
from label_studio_ml.model import LabelStudioMLBase
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import mlflow
import os
import numpy as np
from sklearn.metrics import classification_report

class ActiveLearningNERModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(ActiveLearningNERModel, self).__init__(**kwargs)
        
        # Configuration
        self.base_model = os.environ.get('BASELINE_MODEL_NAME', 'bert-base-cased')
        self.learning_rate = float(os.environ.get('LEARNING_RATE', '3e-5'))
        self.model_dir = os.environ.get('MODEL_DIR', './models')
        self.uncertainty_threshold = float(os.environ.get('UNCERTAINTY_THRESHOLD', '0.7'))
        self.samples_per_iteration = int(os.environ.get('SAMPLES_PER_ITERATION', '500'))
        
        # MLflow setup
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlruns.db')
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment('ner_active_learning')
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = self._load_model()
        
        # Store uncertainty values for active learning
        self.uncertainties = {}

    def _load_model(self):
        """Load or initialize model"""
        # Check if we have a fine-tuned model already
        model_path = os.path.join(self.model_dir, 'best_model')
        if os.path.exists(model_path):
            model = AutoModelForTokenClassification.from_pretrained(model_path)
        else:
            # Initialize from base model
            model = AutoModelForTokenClassification.from_pretrained(
                self.base_model, 
                num_labels=len(self.label_map)
            )
        return model.to(self.device)
    
    def predict(self, tasks, **kwargs):
        """Get predictions and uncertainty scores for tasks"""
        texts = [task['data']['text'] for task in tasks]
        results = []
        
        # Get predictions and uncertainties
        with torch.no_grad():
            self.model.eval()
            for idx, text in enumerate(texts):
                # Tokenize
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding='max_length'
                ).to(self.device)
                
                # Get predictions
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=2)
                predictions = torch.argmax(logits, dim=2)
                
                # Calculate uncertainty (entropy-based)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=2)
                uncertainty = entropy.mean().item()
                
                # Store uncertainty for active learning
                task_id = tasks[idx]['id']
                self.uncertainties[task_id] = uncertainty
                
                # Convert predictions to Label Studio format
                result = self._create_annotation(text, predictions[0], inputs)
                
                results.append({
                    'result': result,
                    'score': 1.0 - uncertainty,
                    'model_version': self.model_version
                })
        
        return results
    
    def fit(self, completions, **kwargs):
        """Train model with active learning"""
        with mlflow.start_run():
            # Extract data from completions
            texts, annotations = [], []
            for completion in completions:
                texts.append(completion['data']['text'])
                annotations.append(completion['annotations'][0]['result'])
            
            # Train the model
            metrics = self._train_model(texts, annotations)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save the model
            self.model.save_pretrained(os.path.join(self.model_dir, 'best_model'))
            self.tokenizer.save_pretrained(os.path.join(self.model_dir, 'best_model'))
            
            return {'status': 'ok', 'metrics': metrics}
    
    def _select_samples_for_annotation(self):
        """Active learning: select most uncertain samples"""
        # Sort tasks by uncertainty
        sorted_tasks = sorted(
            self.uncertainties.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select top N uncertain samples
        selected_tasks = sorted_tasks[:self.samples_per_iteration]
        
        # Return task IDs for annotation
        return [task_id for task_id, _ in selected_tasks]
    
    def _train_model(self, texts, annotations):
        """Train the model"""
        # Convert annotations to training format
        # Training logic with early stopping, etc.
        # Return metrics
        return {
            'train_loss': 0.5,
            'val_f1': 0.8,
            'val_precision': 0.75,
            'val_recall': 0.85
        }
    
    def _create_annotation(self, text, predictions, inputs):
        """Convert model predictions to Label Studio annotation format"""
        # Implementation to convert model outputs to Label Studio format
        # ...
```

## Additional Components to Implement

1. **Data Conversion** - Methods to convert between Label Studio's annotation format and our NER format
2. **Token Alignment** - Logic to align tokenized inputs with model predictions
3. **Model Training** - More sophisticated training routine with early stopping, validation, etc.
4. **Uncertainty Estimation** - Enhanced uncertainty metrics focused on entity boundaries
5. **Sample Selection** - Different strategies for active learning (uncertainty, diversity, hybrid)

## Deployment

1. **Docker Setup** - Create a Dockerfile and docker-compose.yml file
2. **Environment Variables** - Configure the backend with proper settings
3. **Documentation** - Instructions for setup and usage
4. **Testing** - Verify integration with Label Studio

## Key Configuration Options

The backend should support these configuration options via environment variables:

- `BASELINE_MODEL_NAME` - Base pre-trained model (default: 'bert-base-cased')
- `LEARNING_RATE` - Learning rate for fine-tuning (default: 3e-5)
- `MODEL_DIR` - Directory to save model checkpoints (default: './models')
- `UNCERTAINTY_THRESHOLD` - Threshold for selecting samples (default: 0.7)
- `SAMPLES_PER_ITERATION` - Number of samples to select per active learning iteration (default: 500)
- `MLFLOW_TRACKING_URI` - URI for MLflow tracking server (default: 'sqlite:///mlruns.db')

## Usage Workflow

1. Start Label Studio and the ML backend
2. Create a project with NER labeling configuration
3. Connect the ML backend to the project
4. Upload initial ~200 labeled samples
5. Train the initial model
6. Upload a pool of unlabeled data
7. Use the backend to select uncertain samples for annotation
8. Annotate the selected samples
9. Retrain the model
10. Repeat steps 7-9 until reaching desired performance or dataset size

## References

1. [Label Studio ML Backend Repository](https://github.com/HumanSignal/label-studio-ml-backend)
2. [Hugging Face NER Example](https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/examples/huggingface_ner/README.md)
3. [Implementation Guide](./assets/research/active-learning-ner-implementation-guide.md) 