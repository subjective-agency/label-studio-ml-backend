import os
import pathlib
import re
import torch
import numpy as np
import logging
import mlflow
from typing import List, Dict, Optional, Tuple, Any

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, ClassLabel, Value, Sequence, Features
from functools import partial
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Configure logging
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logger.setLevel(getattr(logging, LOG_LEVEL))

# Configure environment variables
MODEL_DIR = os.getenv('MODEL_DIR', './models')
BASELINE_MODEL_NAME = os.getenv('BASELINE_MODEL_NAME', 'dslim/bert-base-NER')
FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME', 'finetuned_model')
LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', '')
START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', '10'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '2e-5'))
NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', '3'))
WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', '0.01'))
MAX_SEQUENCE_LENGTH = int(os.getenv('MAX_SEQUENCE_LENGTH', '128'))
UNCERTAINTY_THRESHOLD = float(os.getenv('UNCERTAINTY_THRESHOLD', '0.8'))
SAMPLES_PER_ITERATION = int(os.getenv('SAMPLES_PER_ITERATION', '500'))
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlruns.db')

# Setup MLflow
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("active_learning_ner")

# Global model instance
_tokenizer = None
_model = None

def get_model_and_tokenizer():
    """Load or initialize tokenizer and model"""
    global _tokenizer, _model
    
    if _tokenizer is None or _model is None:
        try:
            # Try to load finetuned model
            model_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
            logger.info(f"Attempting to load finetuned model from {model_path}")
            
            if os.path.exists(model_path):
                _tokenizer = AutoTokenizer.from_pretrained(model_path)
                _model = AutoModelForTokenClassification.from_pretrained(model_path)
                logger.info(f"Successfully loaded finetuned model from {model_path}")
            else:
                # Load baseline model
                logger.info(f"Loading baseline model {BASELINE_MODEL_NAME}")
                _tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME)
                _model = AutoModelForTokenClassification.from_pretrained(BASELINE_MODEL_NAME)
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fall back to baseline model
            logger.info(f"Falling back to baseline model {BASELINE_MODEL_NAME}")
            _tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME)
            _model = AutoModelForTokenClassification.from_pretrained(BASELINE_MODEL_NAME)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model = _model.to(device)
    
    return _tokenizer, _model

class NERDataset(torch.utils.data.Dataset):
    """Dataset for NER fine-tuning"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Align labels with tokens
        word_ids = encoding.word_ids()
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens
            else:
                try:
                    label_ids.append(int(label[word_idx]))
                except IndexError:
                    # Handle cases where tokenization doesn't align perfectly
                    label_ids.append(-100)
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label_ids)
        }

class ActiveLearningNER(LabelStudioMLBase):
    """Active Learning NER model for Label Studio ML Backend"""
    
    def __init__(self, **kwargs):
        super(ActiveLearningNER, self).__init__(**kwargs)
        
        # Initialize model and tokenizer
        self.tokenizer, self.model = get_model_and_tokenizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store uncertainties for active learning sample selection
        self.uncertainties = {}
        
        # Model configuration
        self.batch_size = 16
        self.max_length = MAX_SEQUENCE_LENGTH
        
        # Initialize version
        self.model_version = self.get('model_version', f'{self.__class__.__name__}-v0.0.1')
        logger.info(f"Model loaded with version: {self.model_version}")
    
    def predict(self, tasks, context=None, **kwargs):
        """Get predictions and uncertainties for tasks"""
        texts = [task['data']['text'] for task in tasks]
        results = []
        
        tokenizer, model = get_model_and_tokenizer()
        model.eval()
        
        li = self.label_interface
        from_name, to_name, value = li.get_first_tag_occurence('Labels', 'Text')
        
        with torch.no_grad():
            for idx, text in enumerate(texts):
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length'
                ).to(self.device)
                
                # Get predictions
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=2)
                predictions = torch.argmax(logits, dim=2)
                
                # Calculate token-level uncertainty (entropy or 1-max prob)
                token_uncertainties = 1 - torch.max(probs, dim=2).values
                mean_uncertainty = token_uncertainties.mean().item()
                
                # Store uncertainty for active learning
                task_id = tasks[idx]['id']
                self.uncertainties[task_id] = mean_uncertainty
                
                # Convert predictions to Label Studio format
                result = self._create_ner_annotation(
                    text, 
                    predictions[0].cpu().numpy(), 
                    token_uncertainties[0].cpu().numpy(),
                    from_name, 
                    to_name
                )
                
                results.append({
                    'result': result,
                    'score': 1.0 - mean_uncertainty,
                    'model_version': self.model_version
                })
                
                logger.debug(f"Prediction for task {task_id}: uncertainty={mean_uncertainty:.4f}")
        
        return ModelResponse(predictions=results, model_version=self.model_version)
    
    def _create_ner_annotation(self, text, predictions, uncertainties, from_name, to_name):
        """Convert model predictions to Label Studio annotation format"""
        tokens = self.tokenizer.tokenize(text)
        token_spans = []
        
        # Calculate token spans in the original text
        start = 0
        for token in text.split():
            end = start + len(token)
            token_spans.append((start, end))
            start = end + 1  # +1 for space
        
        # Group predictions by entity type (accounting for B- and I- prefixes)
        results = []
        id2label = self.model.config.id2label
        
        i = 0
        while i < len(predictions) and i < len(token_spans):
            pred_id = predictions[i]
            pred_label = id2label.get(pred_id, 'O')
            uncertainty = uncertainties[i]
            
            # Skip non-entity tokens
            if pred_label == 'O' or pred_label == -100:
                i += 1
                continue
                
            # Handle B- prefix (beginning of entity)
            if pred_label.startswith('B-'):
                entity_label = pred_label[2:]  # Remove B- prefix
                start_token = i
                end_token = i
                
                # Find consecutive I- tokens of the same entity type
                i += 1
                while (i < len(predictions) and 
                       id2label.get(predictions[i], 'O').startswith('I-') and 
                       id2label.get(predictions[i], 'O')[2:] == entity_label):
                    end_token = i
                    i += 1
                
                # Create entity annotation
                results.append({
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'labels',
                    'value': {
                        'start': token_spans[start_token][0],
                        'end': token_spans[end_token][1],
                        'labels': [entity_label]
                    },
                    'score': float(1.0 - uncertainty)
                })
            else:
                # Skip any unexpected formats
                i += 1
        
        return results
    
    def select_uncertain_samples(self, task_ids):
        """Select most uncertain samples for annotation"""
        if not self.uncertainties:
            return []
            
        # Sort tasks by uncertainty
        sorted_tasks = sorted(
            [(task_id, uncertainty) for task_id, uncertainty in self.uncertainties.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter to include only unlabeled tasks
        unlabeled_tasks = [t[0] for t in sorted_tasks if t[0] in task_ids]
        
        # Select top N uncertain samples
        selected_count = min(SAMPLES_PER_ITERATION, len(unlabeled_tasks))
        selected_tasks = unlabeled_tasks[:selected_count]
        
        logger.info(f"Selected {len(selected_tasks)} uncertain samples for annotation")
        return selected_tasks
    
    def fit(self, event, data, **kwargs):
        """Train model with active learning"""
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"Skip training: event {event} is not supported")
            return
        
        logger.info(f"Starting model training: event={event}")
        
        # Get annotations from Label Studio
        try:
            annotations = self._get_annotations_from_event(data)
            
            if not annotations or len(annotations) < START_TRAINING_EACH_N_UPDATES:
                logger.info(f"Not enough annotations for training: {len(annotations)} < {START_TRAINING_EACH_N_UPDATES}")
                return
            
            logger.info(f"Training with {len(annotations)} annotations")
            
            # Start MLflow run
            with mlflow.start_run():
                # Prepare dataset for training
                train_dataset = self._prepare_training_dataset(annotations)
                
                # Train the model
                metrics = self._train_model(train_dataset)
                
                # Log metrics to MLflow
                mlflow.log_metrics(metrics)
                
                # Save the model
                save_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                logger.info(f"Model saved to {save_path}")
                
                # Update model version
                self.model_version = f"{self.__class__.__name__}-v{metrics['val_f1']:.4f}"
                self.set("model_version", self.model_version)
                
                return {
                    'status': 'ok',
                    'metrics': metrics,
                    'model_version': self.model_version
                }
        
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}
    
    def _get_annotations_from_event(self, data):
        """Extract annotations from event data"""
        annotation = data['annotation']
        project_id = annotation['project']
        
        # TODO: Implement this to get all annotations from the project
        # For now, just return the single annotation from the event
        result = []
        if annotation.get('result'):
            result.append({
                'text': self.preload_task_data(data['task'], data['task']['data']['text']),
                'result': annotation['result']
            })
        
        return result
    
    def _prepare_training_dataset(self, annotations):
        """Convert annotations to training dataset"""
        no_label = 'O'
        label_to_id = {no_label: 0}
        
        # Extract all unique labels
        for annotation in annotations:
            for item in annotation.get('result', []):
                if item.get('value', {}).get('labels'):
                    label = item['value']['labels'][0]
                    if f"B-{label}" not in label_to_id:
                        label_to_id[f"B-{label}"] = len(label_to_id)
                    if f"I-{label}" not in label_to_id:
                        label_to_id[f"I-{label}"] = len(label_to_id)
        
        logger.debug(f"Label map: {label_to_id}")
        
        # Convert annotations to NER format
        train_items = []
        for annotation in annotations:
            text = annotation['text']
            result = annotation.get('result', [])
            
            # Convert to token-level labels
            tokens = text.split()
            labels = [0] * len(tokens)  # Default to 'O'
            
            for item in result:
                if item.get('type') != 'labels':
                    continue
                
                value = item.get('value', {})
                if not value.get('labels'):
                    continue
                
                entity_label = value['labels'][0]
                start_pos = value.get('start', 0)
                end_pos = value.get('end', 0)
                
                # Find token indices corresponding to character positions
                start_token = 0
                end_token = 0
                pos = 0
                
                for i, token in enumerate(tokens):
                    token_start = pos
                    token_end = pos + len(token)
                    
                    if token_start <= start_pos < token_end:
                        start_token = i
                    if token_start < end_pos <= token_end:
                        end_token = i
                        break
                    
                    pos = token_end + 1  # +1 for space
                
                # Assign B- to first token, I- to rest
                for i in range(start_token, end_token + 1):
                    if i == start_token:
                        labels[i] = label_to_id.get(f"B-{entity_label}", 0)
                    else:
                        labels[i] = label_to_id.get(f"I-{entity_label}", 0)
            
            train_items.append({
                'tokens': tokens,
                'labels': labels
            })
        
        # Create features for dataset
        features = Features({
            'tokens': Sequence(Value('string')),
            'labels': Sequence(ClassLabel(names=list(label_to_id.keys())))
        })
        
        # Create dataset
        dataset = Dataset.from_list(train_items, features=features)
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            partial(self._tokenize_and_align_labels, tokenizer=self.tokenizer),
            batched=True
        )
        
        return tokenized_dataset
    
    def _tokenize_and_align_labels(self, examples, tokenizer):
        """Tokenize and align labels with wordpiece tokens"""
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length"
        )

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
                
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def _train_model(self, dataset):
        """Train the model with the prepared dataset"""
        # Create data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(pathlib.Path(MODEL_DIR) / "checkpoints"),
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            evaluation_strategy="no",
            save_strategy="no",
            logging_dir="./logs",
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Calculate metrics
        train_loss = train_result.metrics["train_loss"]
        
        # Evaluate on the same dataset (for simplicity)
        # In a more complete implementation, you would use a separate validation set
        eval_metrics = self._evaluate_model(dataset)
        
        metrics = {
            "train_loss": train_loss,
            "val_f1": eval_metrics["val_f1"],
            "val_precision": eval_metrics["val_precision"],
            "val_recall": eval_metrics["val_recall"]
        }
        
        return metrics
    
    def _evaluate_model(self, dataset):
        """Evaluate the model on a dataset"""
        self.model.eval()
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        # Get predictions
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=2)
                
                # Filter out padding tokens (-100)
                for i in range(len(input_ids)):
                    valid_preds = []
                    valid_labels = []
                    
                    for j in range(len(preds[i])):
                        if labels[i][j] != -100:
                            valid_preds.append(preds[i][j].item())
                            valid_labels.append(labels[i][j].item())
                    
                    all_preds.extend(valid_preds)
                    all_labels.extend(valid_labels)
        
        # Calculate metrics
        metrics = classification_report(
            all_labels,
            all_preds,
            output_dict=True,
            zero_division=0
        )
        
        return {
            "val_loss": total_loss / len(dataloader),
            "val_f1": metrics["weighted avg"]["f1-score"],
            "val_precision": metrics["weighted avg"]["precision"],
            "val_recall": metrics["weighted avg"]["recall"]
        }
    
    def get_entity_boundary_uncertainty(self, predictions, uncertainties):
        """Calculate enhanced uncertainty focusing on entity boundaries"""
        boundary_uncertainties = []
        
        for text_preds, text_uncs in zip(predictions, uncertainties):
            # Higher weight for tokens at entity boundaries
            boundary_weight = 1.5
            weighted_uncs = []
            
            for i, (pred, unc) in enumerate(zip(text_preds, text_uncs)):
                # Check if token is at entity boundary
                is_boundary = False
                
                if i > 0 and text_preds[i-1] != pred:
                    is_boundary = True
                
                if i < len(text_preds)-1 and text_preds[i+1] != pred:
                    is_boundary = True
                
                # Apply weight to uncertainty
                weighted_uncs.append(unc * (boundary_weight if is_boundary else 1.0))
            
            # Calculate mean weighted uncertainty
            boundary_uncertainties.append(np.mean(weighted_uncs) if weighted_uncs else 0)
        
        return boundary_uncertainties 