#!/usr/bin/env python3
"""
Utility functions for active learning workflow with Label Studio
"""

import argparse
import json
import requests
import sys
import os
import logging
from pprint import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Active Learning NER Utilities')
    
    # Main command
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Upload initial data
    upload_parser = subparsers.add_parser('upload-initial', help='Upload initial labeled data')
    upload_parser.add_argument('--url', type=str, default='http://localhost:8080', help='Label Studio URL')
    upload_parser.add_argument('--token', type=str, required=True, help='Label Studio API token')
    upload_parser.add_argument('--project', type=int, required=True, help='Project ID')
    upload_parser.add_argument('--file', type=str, default='data/samples/initial_data.jsonl', help='Path to JSONL file with initial data')
    
    # Upload unlabeled data
    upload_unlabeled_parser = subparsers.add_parser('upload-unlabeled', help='Upload unlabeled data')
    upload_unlabeled_parser.add_argument('--url', type=str, default='http://localhost:8080', help='Label Studio URL')
    upload_unlabeled_parser.add_argument('--token', type=str, required=True, help='Label Studio API token')
    upload_unlabeled_parser.add_argument('--project', type=int, required=True, help='Project ID')
    upload_unlabeled_parser.add_argument('--file', type=str, default='data/samples/unlabeled_data.jsonl', help='Path to JSONL file with unlabeled data')
    
    # Select samples
    select_parser = subparsers.add_parser('select-samples', help='Select uncertain samples for annotation')
    select_parser.add_argument('--url', type=str, default='http://localhost:9090', help='ML backend URL')
    select_parser.add_argument('--project', type=int, required=True, help='Project ID')
    select_parser.add_argument('--count', type=int, default=10, help='Number of samples to select')
    
    # Start training
    train_parser = subparsers.add_parser('start-training', help='Start model training')
    train_parser.add_argument('--url', type=str, default='http://localhost:8080', help='Label Studio URL')
    train_parser.add_argument('--token', type=str, required=True, help='Label Studio API token')
    train_parser.add_argument('--project', type=int, required=True, help='Project ID')
    
    return parser.parse_args()

def upload_initial_data(url, token, project_id, file_path):
    """Upload initial labeled data to Label Studio"""
    logger.info(f"Uploading initial labeled data from {file_path} to project {project_id}")
    
    # Load data from JSONL file
    tasks = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Convert to Label Studio format
            text = data['text']
            labels = data['labels']
            words = text.split()
            
            annotations = []
            current_entity = None
            start_offset = 0
            
            for i, (word, label) in enumerate(zip(words, labels)):
                if label.startswith('B-'):
                    # End previous entity if exists
                    if current_entity:
                        annotations.append(current_entity)
                    
                    # Start new entity
                    entity_type = label[2:] # Remove 'B-' prefix
                    start_char = text.find(word, start_offset)
                    end_char = start_char + len(word)
                    
                    current_entity = {
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": start_char,
                            "end": end_char,
                            "text": word,
                            "labels": [entity_type]
                        }
                    }
                    
                    # Update start offset for next search
                    start_offset = end_char
                
                elif label.startswith('I-') and current_entity:
                    # Continue current entity
                    entity_type = label[2:] # Remove 'I-' prefix
                    
                    # Check if entity type matches the current entity
                    if entity_type == current_entity['value']['labels'][0]:
                        # Find word in text
                        start_char = text.find(word, start_offset)
                        end_char = start_char + len(word)
                        
                        # Update entity end position
                        current_entity['value']['end'] = end_char
                        current_entity['value']['text'] += f" {word}"
                        
                        # Update start offset
                        start_offset = end_char
                    else:
                        # Entity type changed, end current entity and start new one
                        annotations.append(current_entity)
                        current_entity = None
                        
                        # Start new entity
                        start_char = text.find(word, start_offset)
                        end_char = start_char + len(word)
                        
                        current_entity = {
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                            "value": {
                                "start": start_char,
                                "end": end_char,
                                "text": word,
                                "labels": [entity_type]
                            }
                        }
                        
                        # Update start offset
                        start_offset = end_char
                else:
                    # End current entity if exists
                    if current_entity:
                        annotations.append(current_entity)
                        current_entity = None
                    
                    # Update start offset for next search
                    start_offset += len(word) + 1  # +1 for space
            
            # Add final entity if exists
            if current_entity:
                annotations.append(current_entity)
            
            # Create task with annotations
            tasks.append({
                "data": {
                    "text": text
                },
                "annotations": [{
                    "result": annotations
                }]
            })
    
    # Upload to Label Studio
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{url}/api/projects/{project_id}/import",
        headers=headers,
        json=tasks
    )
    
    if response.status_code == 201:
        logger.info(f"Successfully uploaded {len(tasks)} tasks with annotations")
        return True
    else:
        logger.error(f"Failed to upload data: {response.status_code} - {response.text}")
        return False

def upload_unlabeled_data(url, token, project_id, file_path):
    """Upload unlabeled data to Label Studio"""
    logger.info(f"Uploading unlabeled data from {file_path} to project {project_id}")
    
    # Load data from JSONL file
    tasks = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Convert to Label Studio format
            tasks.append({
                "data": {
                    "text": data['text']
                }
            })
    
    # Upload to Label Studio
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{url}/api/projects/{project_id}/import",
        headers=headers,
        json=tasks
    )
    
    if response.status_code == 201:
        logger.info(f"Successfully uploaded {len(tasks)} unlabeled tasks")
        return True
    else:
        logger.error(f"Failed to upload data: {response.status_code} - {response.text}")
        return False

def select_uncertain_samples(url, project_id, count):
    """Select uncertain samples for annotation using ML backend"""
    logger.info(f"Selecting {count} uncertain samples from project {project_id}")
    
    # Call ML backend to select samples
    data = {
        "project_id": project_id,
        "limit": count
    }
    
    try:
        response = requests.post(f"{url}/select_samples", json=data)
        result = response.json()
        
        logger.info(f"Selected {len(result.get('selected_ids', []))} samples for annotation")
        return result
    except Exception as e:
        logger.error(f"Error selecting samples: {e}")
        return {"error": str(e)}

def start_training(url, token, project_id):
    """Start model training manually"""
    logger.info(f"Starting model training for project {project_id}")
    
    # Get ML backend ID
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json"
    }
    
    # Get project MLBackends
    response = requests.get(
        f"{url}/api/projects/{project_id}/backends",
        headers=headers
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to get project backends: {response.status_code} - {response.text}")
        return False
    
    backends = response.json()
    if not backends:
        logger.error("No ML backends connected to this project")
        return False
    
    # Start training for the first backend
    backend_id = backends[0]['id']
    
    response = requests.post(
        f"{url}/api/ml/{backend_id}/train",
        headers=headers
    )
    
    if response.status_code == 200:
        logger.info("Successfully started model training")
        return True
    else:
        logger.error(f"Failed to start training: {response.status_code} - {response.text}")
        return False

def main():
    args = parse_args()
    
    try:
        if args.command == 'upload-initial':
            upload_initial_data(args.url, args.token, args.project, args.file)
            
        elif args.command == 'upload-unlabeled':
            upload_unlabeled_data(args.url, args.token, args.project, args.file)
            
        elif args.command == 'select-samples':
            result = select_uncertain_samples(args.url, args.project, args.count)
            pprint(result)
            
        elif args.command == 'start-training':
            start_training(args.url, args.token, args.project)
            
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
        
    sys.exit(0)

if __name__ == "__main__":
    main() 