#!/usr/bin/env python3
# /// script
# requires-python = "==3.11"
# dependencies = ["torch", "transformers", "mlflow", "pydantic-settings", "scikit-learn", "numpy", "typer"]
# ///

import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import the active learning components
from wapaganda.active_learning_ner.core import ActiveLearningPipeline
from wapaganda.active_learning.settings import ActiveLearningSettings

console = Console()
app = typer.Typer(help="Active Learning NER Pipeline")

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_texts_and_labels(data: List[Dict[str, Any]]) -> tuple[List[str], List[List[str]]]:
    """Extract texts and labels from the data."""
    texts = []
    labels = []
    
    for item in data:
        texts.append(item.get('text', ''))
        labels.append(item.get('labels', []))
        
    return texts, labels

@app.command()
def train_initial_model(
    initial_data_path: str = typer.Argument(..., help="Path to initial labeled data (JSONL)"),
    project_name: str = typer.Option("ner-active-learning", help="Name of the Label Studio project"),
    description: str = typer.Option("NER Active Learning Project", help="Project description"),
    mlflow_uri: Optional[str] = typer.Option(None, help="MLflow tracking URI"),
    output_dir: str = typer.Option("./active_learning_output", help="Directory to save output files")
):
    """
    Phase 2, Step 4: Train the initial NER model with ~200 manually labeled samples.
    """
    console.print(f"[bold blue]Phase 2, Step 4: Initial Model Training[/bold blue]")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load initial data
    with console.status("Loading initial data..."):
        if not os.path.exists(initial_data_path):
            console.print(f"[bold red]Error: File {initial_data_path} not found![/bold red]")
            return

        data = load_jsonl(initial_data_path)
        if len(data) < 10:  # Sanity check
            console.print(f"[bold yellow]Warning: Only {len(data)} samples found. Recommended: 200+[/bold yellow]")
            
        texts, labels = extract_texts_and_labels(data)
        console.print(f"Loaded {len(texts)} samples from {initial_data_path}")
    
    # Configure settings
    settings = ActiveLearningSettings()
    if mlflow_uri:
        settings.mlflow_tracking_uri = mlflow_uri
    
    # Initialize pipeline
    with console.status("Initializing Active Learning Pipeline..."):
        pipeline = ActiveLearningPipeline(settings)
    
    # Initialize project and train initial model
    with console.status("Training initial model... This may take some time..."):
        project_id = pipeline.initialize_project(
            project_name=project_name,
            description=description,
            initial_texts=texts,
            initial_labels=labels
        )
        
        # Save project ID
        with open(os.path.join(output_dir, "project_info.json"), "w") as f:
            json.dump({
                "project_id": project_id,
                "project_name": project_name
            }, f, indent=2)
    
    console.print(f"[bold green]Success![/bold green] Initial model trained and project initialized.")
    console.print(f"Project ID: {project_id}")
    console.print(f"Project information saved to: {os.path.join(output_dir, 'project_info.json')}")
    console.print("\nNext steps:")
    console.print("1. Run active learning iterations to select uncertain samples")
    console.print("2. Annotate selected samples in Label Studio")
    console.print("3. Update the model with new annotations")

@app.command()
def select_samples(
    project_info_path: str = typer.Argument(..., help="Path to project_info.json"),
    unlabeled_data_path: str = typer.Argument(..., help="Path to unlabeled data (JSONL)"),
    output_path: str = typer.Option("selected_samples.jsonl", help="Path to save selected samples"),
    mlflow_uri: Optional[str] = typer.Option(None, help="MLflow tracking URI")
):
    """
    Phase 3, Step 5: Select uncertain samples for annotation.
    """
    console.print(f"[bold blue]Phase 3, Step 5: Uncertainty-Based Sample Selection[/bold blue]")
    
    # Load project info
    with console.status("Loading project information..."):
        if not os.path.exists(project_info_path):
            console.print(f"[bold red]Error: File {project_info_path} not found![/bold red]")
            return
            
        with open(project_info_path, "r") as f:
            project_info = json.load(f)
            project_id = project_info.get("project_id")
            
        if not project_id:
            console.print("[bold red]Error: Project ID not found in project_info.json![/bold red]")
            return
    
    # Load unlabeled data
    with console.status("Loading unlabeled data..."):
        if not os.path.exists(unlabeled_data_path):
            console.print(f"[bold red]Error: File {unlabeled_data_path} not found![/bold red]")
            return
            
        data = load_jsonl(unlabeled_data_path)
        unlabeled_texts = [item.get('text', '') for item in data]
        console.print(f"Loaded {len(unlabeled_texts)} unlabeled samples")
    
    # Configure settings
    settings = ActiveLearningSettings()
    if mlflow_uri:
        settings.mlflow_tracking_uri = mlflow_uri
    
    # Initialize pipeline
    with console.status("Initializing Active Learning Pipeline..."):
        pipeline = ActiveLearningPipeline(settings)
    
    # Run active learning iteration
    with console.status("Selecting uncertain samples... This may take some time..."):
        results = pipeline.run_active_learning_iteration(
            project_id=project_id,
            unlabeled_texts=unlabeled_texts
        )
        
        selected_texts = results.get("selected_texts", [])
        selected_indices = results.get("selected_indices", [])
        
        console.print(f"Selected {len(selected_texts)} samples for annotation")
        
        # Save selected samples
        with open(output_path, "w") as f:
            for idx in selected_indices:
                f.write(json.dumps(data[idx]) + "\n")
    
    console.print(f"[bold green]Success![/bold green] Selected samples saved to: {output_path}")
    console.print("\nNext steps:")
    console.print("1. Annotate the selected samples in Label Studio")
    console.print("2. Update the model with the new annotations")

@app.command()
def update_model(
    project_info_path: str = typer.Argument(..., help="Path to project_info.json"),
    current_data_path: Optional[str] = typer.Option(None, help="Path to current labeled data (JSONL)"),
    mlflow_uri: Optional[str] = typer.Option(None, help="MLflow tracking URI")
):
    """
    Phase 3, Step 7: Retrain the model with newly annotated data.
    """
    console.print(f"[bold blue]Phase 3, Step 7: Model Retraining and Evaluation[/bold blue]")
    
    # Load project info
    with console.status("Loading project information..."):
        if not os.path.exists(project_info_path):
            console.print(f"[bold red]Error: File {project_info_path} not found![/bold red]")
            return
            
        with open(project_info_path, "r") as f:
            project_info = json.load(f)
            project_id = project_info.get("project_id")
            
        if not project_id:
            console.print("[bold red]Error: Project ID not found in project_info.json![/bold red]")
            return
    
    # Load current data if provided
    current_texts = None
    current_labels = None
    if current_data_path:
        with console.status("Loading current labeled data..."):
            if not os.path.exists(current_data_path):
                console.print(f"[bold red]Error: File {current_data_path} not found![/bold red]")
                return
                
            data = load_jsonl(current_data_path)
            current_texts, current_labels = extract_texts_and_labels(data)
            console.print(f"Loaded {len(current_texts)} current labeled samples")
    
    # Configure settings
    settings = ActiveLearningSettings()
    if mlflow_uri:
        settings.mlflow_tracking_uri = mlflow_uri
    
    # Initialize pipeline
    with console.status("Initializing Active Learning Pipeline..."):
        pipeline = ActiveLearningPipeline(settings)
    
    # Update model
    with console.status("Updating model with new annotations... This may take some time..."):
        metrics = pipeline.update_model(
            project_id=project_id,
            current_texts=current_texts,
            current_labels=current_labels
        )
        
        if not metrics:
            console.print("[bold yellow]Warning: No new annotations found. Model not updated.[/bold yellow]")
            return
            
        console.print("\n[bold green]Model Updated Successfully![/bold green]")
        console.print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            console.print(f"  {metric}: {value:.4f}")
    
    console.print("\nNext steps:")
    console.print("1. Run another active learning iteration")
    console.print("2. Continue the annotation-retraining loop until desired performance")

if __name__ == "__main__":
    app() 