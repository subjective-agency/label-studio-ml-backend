from typing import List, Dict, Any, Tuple
import mlflow
from wapaganda.active_learning.settings import ActiveLearningSettings
from wapaganda.active_learning.model_trainer import ModelTrainer
from wapaganda.active_learning.uncertainty_estimator import UncertaintyEstimator
from wapaganda.active_learning.label_studio_connector import LabelStudioConnector


class ActiveLearningPipeline:
    def __init__(self, settings: ActiveLearningSettings | None = None):
        self.settings = settings or ActiveLearningSettings()
        self.model_trainer = ModelTrainer(self.settings)
        self.uncertainty_estimator = UncertaintyEstimator(self.settings)
        self.label_studio = LabelStudioConnector(self.settings)
        
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)
        
    def initialize_project(
        self,
        project_name: str,
        description: str | None = None,
        initial_texts: List[str] | None = None,
        initial_labels: List[List[str]] | None = None
    ) -> int:
        """
        Initialize a new active learning project.
        
        Args:
            project_name: Name of the Label Studio project
            description: Project description
            initial_texts: Initial labeled texts for training
            initial_labels: Initial labels for training
            
        Returns:
            Project ID
        """
        project_id = self.label_studio.create_project(project_name, description)
        
        if initial_texts and initial_labels:
            # Train initial model
            self.model_trainer.train(initial_texts, initial_labels)
            
            # Upload initial data to Label Studio
            self.label_studio.upload_tasks(project_id, initial_texts)
            
        return project_id
        
    def run_active_learning_iteration(
        self,
        project_id: int,
        unlabeled_texts: List[str],
        current_texts: List[str] | None = None,
        current_labels: List[List[str]] | None = None
    ) -> Dict[str, Any]:
        """
        Run one iteration of active learning.
        
        Args:
            project_id: Label Studio project ID
            unlabeled_texts: Pool of unlabeled texts
            current_texts: Currently labeled texts
            current_labels: Current labels
            
        Returns:
            Dictionary with iteration results
        """
        with mlflow.start_run(nested=True):
            # Get model predictions and uncertainties
            predictions, uncertainties = self.model_trainer.predict(unlabeled_texts)
            
            # Select uncertain samples
            selected_texts, selected_indices = self.uncertainty_estimator.select_uncertain_samples(
                unlabeled_texts,
                uncertainties
            )
            
            # Calculate boundary-aware uncertainty
            boundary_uncertainties = self.uncertainty_estimator.get_entity_boundary_uncertainty(
                predictions,
                uncertainties
            )
            
            # Upload selected samples to Label Studio
            self.label_studio.upload_tasks(
                project_id,
                selected_texts,
                [predictions[i] for i in selected_indices],
                [uncertainties[i] for i in selected_indices]
            )
            
            mlflow.log_metrics({
                "mean_uncertainty": sum(boundary_uncertainties) / len(boundary_uncertainties),
                "max_uncertainty": max(boundary_uncertainties),
                "selected_samples": len(selected_texts)
            })
            
            return {
                "selected_indices": selected_indices,
                "selected_texts": selected_texts,
                "uncertainties": [uncertainties[i] for i in selected_indices],
                "predictions": [predictions[i] for i in selected_indices]
            }
            
    def update_model(
        self,
        project_id: int,
        current_texts: List[str] | None = None,
        current_labels: List[List[str]] | None = None
    ) -> Dict[str, float]:
        """
        Update the model with newly labeled data.
        
        Args:
            project_id: Label Studio project ID
            current_texts: Optional list of current training texts
            current_labels: Optional list of current training labels
            
        Returns:
            Training metrics
        """
        with mlflow.start_run(nested=True):
            # Get new annotations from Label Studio
            annotations = self.label_studio.get_annotations(project_id)
            
            if not annotations:
                return {}
                
            # Combine new annotations with existing data
            texts = [a["text"] for a in annotations]
            labels = [a["labels"] for a in annotations]
            
            if current_texts and current_labels:
                texts.extend(current_texts)
                labels.extend(current_labels)
                
            # Train model on updated dataset
            metrics = self.model_trainer.train(texts, labels)
            
            return metrics
