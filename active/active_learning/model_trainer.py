from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
import numpy as np
from sklearn.metrics import classification_report
import mlflow
from wapaganda.active_learning.settings import ActiveLearningSettings

class NERDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer: Any, max_length: int):
        settings = ActiveLearningSettings()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {label: i for i, label in enumerate(settings.entity_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
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
                label_ids.append(-100)
            else:
                label_ids.append(self.label2id[label[word_idx]])
                
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label_ids)
        }

class ModelTrainer:
    def __init__(self, settings: ActiveLearningSettings):
        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            settings.base_model_name,
            num_labels=len(settings.entity_labels),
            id2label={i: label for i, label in enumerate(settings.entity_labels)},
            label2id={label: i for i, label in enumerate(settings.entity_labels)}
        )
        self.model.to(settings.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
    def train(
        self, 
        train_texts: List[str], 
        train_labels: List[List[str]], 
        val_texts: List[str] | None = None, 
        val_labels: List[List[str]] | None = None
    ) -> Dict[str, float]:
        train_dataset = NERDataset(
            train_texts, 
            train_labels, 
            self.tokenizer, 
            self.settings.max_sequence_length
        )
        train_loader = DataLoader(train_dataset, batch_size=self.settings.batch_size, shuffle=True)
        
        if val_texts and val_labels:
            val_dataset = NERDataset(
                val_texts, 
                val_labels, 
                self.tokenizer, 
                self.settings.max_sequence_length
            )
            val_loader = DataLoader(val_dataset, batch_size=self.settings.batch_size)
        
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(self.settings.device)
            attention_mask = batch["attention_mask"].to(self.settings.device)
            labels = batch["labels"].to(self.settings.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
        metrics = {"train_loss": total_loss / len(train_loader)}
        
        if val_texts and val_labels:
            val_metrics = self.evaluate(val_texts, val_labels)
            metrics.update(val_metrics)
            
        mlflow.log_metrics(metrics)
        return metrics
        
    def evaluate(self, texts: List[str], labels: List[List[str]]) -> Dict[str, float]:
        dataset = NERDataset(texts, labels, self.tokenizer, self.settings.max_sequence_length)
        dataloader = DataLoader(dataset, batch_size=self.settings.batch_size)
        
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.settings.device)
                attention_mask = batch["attention_mask"].to(self.settings.device)
                labels = batch["labels"].to(self.settings.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=2)
                
                # Remove padding (-100)
                valid_preds = preds[labels != -100]
                valid_labels = labels[labels != -100]
                
                all_preds.extend(valid_preds.cpu().numpy())
                all_labels.extend(valid_labels.cpu().numpy())
        
        metrics = classification_report(
            all_labels,
            all_preds,
            output_dict=True,
            zero_division=0,
            target_names=self.settings.entity_labels
        )
        
        return {
            "val_loss": total_loss / len(dataloader),
            "val_f1": metrics["weighted avg"]["f1-score"],
            "val_precision": metrics["weighted avg"]["precision"],
            "val_recall": metrics["weighted avg"]["recall"]
        }
        
    def predict(self, texts: List[str]) -> Tuple[List[List[str]], List[List[float]]]:
        dataset = NERDataset(
            texts,
            [[self.settings.entity_labels[0]] * len(text.split()) for text in texts],
            self.tokenizer,
            self.settings.max_sequence_length
        )
        dataloader = DataLoader(dataset, batch_size=self.settings.batch_size)
        
        self.model.eval()
        all_predictions: List[List[str]] = []
        all_uncertainties: List[List[float]] = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.settings.device)
                attention_mask = batch["attention_mask"].to(self.settings.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=2)
                uncertainties = 1 - torch.max(probs, dim=2).values
                predictions = torch.argmax(logits, dim=2)
                
                for i in range(len(input_ids)):
                    valid_preds = []
                    valid_uncertainties = []
                    word_ids = self.tokenizer(
                        texts[i],
                        max_length=self.settings.max_sequence_length,
                        padding="max_length",
                        truncation=True
                    ).word_ids()
                    
                    current_word_idx = None
                    for j, word_idx in enumerate(word_ids):
                        if word_idx is None or word_idx == current_word_idx:
                            continue
                        current_word_idx = word_idx
                        valid_preds.append(self.settings.entity_labels[predictions[i][j]])
                        valid_uncertainties.append(uncertainties[i][j].item())
                    
                    all_predictions.append(valid_preds)
                    all_uncertainties.append(valid_uncertainties)
        
        return all_predictions, all_uncertainties

# settings = ActiveLearningSettings() 