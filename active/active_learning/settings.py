from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os

class ActiveLearningSettings(BaseSettings):
    """Configuration for the active learning pipeline"""
    # MLflow settings
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    mlflow_experiment_name: str = "active_learning_ner"
    
    # Model settings
    base_model_name: str = "bert-base-cased"  # or "roberta-base"
    device: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Data settings
    data_dir: str = "data"
    models_dir: str = "models"
    initial_train_size: int = 200
    validation_size: int = 40
    
    # Active learning settings
    batch_size: int = 32
    uncertainty_threshold: float = 0.8
    max_sequence_length: int = 128
    samples_per_iteration: int = 500
    
    # Label Studio settings
    label_studio_url: str = "http://localhost:17777"
    label_studio_token: str = ""
    
    # Entity labels
    entity_labels: List[str] = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-GEO", "I-GEO", "B-POS", "I-POS", "B-ART", "I-ART", "B-EVENT", "I-EVENT"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    ) 