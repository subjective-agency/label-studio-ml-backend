from typing import List, Dict, Any
from label_studio_sdk import Client
import json
import numpy as np
from wapaganda.active_learning.settings import ActiveLearningSettings

class LabelStudioConnector:
    def __init__(self, settings: ActiveLearningSettings):
        self.settings = settings
        self.client = Client(
            url=settings.label_studio_url,
            api_key=settings.label_studio_token
        )
        
    def create_project(
        self,
        name: str,
        description: str | None = None
    ) -> int:
        """
        Create a new Label Studio project for NER annotation.
        
        Args:
            name: Project name
            description: Project description
            
        Returns:
            Project ID
        """
        label_config = self._generate_label_config()
        
        project = self.client.start_project(
            title=name,
            description=description or "",
            label_config=label_config
        )
        
        return project.id
        
    def upload_tasks(
        self,
        project_id: int,
        texts: List[str],
        predictions: List[List[str]] | None = None,
        uncertainties: List[List[float]] | None = None
    ) -> None:
        """
        Upload texts as tasks to Label Studio project.
        
        Args:
            project_id: Label Studio project ID
            texts: List of texts to upload
            predictions: Optional model predictions for pre-annotation
            uncertainties: Optional uncertainty scores for each token
        """
        project = self.client.get_project(project_id)
        tasks = []
        
        for i, text in enumerate(texts):
            task = {
                "data": {
                    "text": text
                }
            }
            
            if predictions and uncertainties:
                task["predictions"] = [{
                    "model_version": "1",
                    "score": 1.0 - float(np.mean(uncertainties[i])),
                    "result": self._format_predictions(text, predictions[i], uncertainties[i])
                }]
                
            tasks.append(task)
            
        project.import_tasks(tasks)
        
    def get_annotations(
        self,
        project_id: int,
        task_ids: List[int] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Get completed annotations from Label Studio.
        
        Args:
            project_id: Label Studio project ID
            task_ids: Optional list of specific task IDs to fetch
            
        Returns:
            List of annotations with text and labels
        """
        project = self.client.get_project(project_id)
        tasks = project.get_tasks()
        
        if task_ids:
            tasks = [t for t in tasks if t["id"] in task_ids]
            
        annotations = []
        
        for task in tasks:
            if not task.get("annotations"):
                continue
                
            text = task["data"]["text"]
            annotation = task["annotations"][0]  # Get first annotation
            result = annotation.get("result", [])
            
            labels = self._parse_annotation_result(text, result)
            annotations.append({
                "text": text,
                "labels": labels,
                "task_id": task["id"]
            })
            
        return annotations
        
    def _generate_label_config(self) -> str:
        """Generate Label Studio config XML for NER labeling."""
        label_items = "\n".join(
            f'<Label value="{label}" background="#{hash(label)[:6]}" />'
            for label in self.settings.entity_labels if label != "O"
        )
        
        return f"""
        <View>
            <Labels name="label" toName="text">
                {label_items}
            </Labels>
            <Text name="text" value="$text" />
        </View>
        """
        
    def _format_predictions(
        self,
        text: str,
        predictions: List[str],
        uncertainties: List[float]
    ) -> List[Dict[str, Any]]:
        """Format predictions for Label Studio import."""
        words = text.split()
        results = []
        current_entity = None
        start = 0
        
        for word, pred, unc in zip(words, predictions, uncertainties):
            if pred == "O":
                if current_entity:
                    results.append(current_entity)
                    current_entity = None
            else:
                label = pred[2:]  # Remove B- or I- prefix
                if pred.startswith("B-") or not current_entity:
                    if current_entity:
                        results.append(current_entity)
                    current_entity = {
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": start + len(word),
                            "text": word,
                            "labels": [label]
                        },
                        "score": float(1.0 - unc)
                    }
                else:
                    current_entity["value"]["end"] = start + len(word)
                    current_entity["value"]["text"] += f" {word}"
                    current_entity["score"] = min(
                        current_entity["score"],
                        float(1.0 - unc)
                    )
                    
            start += len(word) + 1  # +1 for space
            
        if current_entity:
            results.append(current_entity)
            
        return results
        
    def _parse_annotation_result(
        self,
        text: str,
        result: List[Dict[str, Any]]
    ) -> List[str]:
        """Parse Label Studio annotation result into token labels."""
        words = text.split()
        labels = ["O"] * len(words)
        
        for annotation in result:
            if annotation["type"] != "labels":
                continue
                
            value = annotation["value"]
            label = value["labels"][0]
            start_word = len(text[:value["start"]].split())
            end_word = len(text[:value["end"]].split())
            
            for i in range(start_word, end_word):
                if i == start_word:
                    labels[i] = f"B-{label}"
                else:
                    labels[i] = f"I-{label}"
                    
        return labels

# settings = ActiveLearningSettings() 