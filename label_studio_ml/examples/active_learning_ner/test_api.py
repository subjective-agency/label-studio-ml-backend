#!/usr/bin/env python3
"""
Test script for Active Learning NER backend API
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
    parser = argparse.ArgumentParser(description='Test Active Learning NER backend API')
    parser.add_argument('--url', dest='url', type=str, default='http://localhost:9090',
                        help='URL of the backend server')
    parser.add_argument('--action', dest='action', type=str, default='health',
                        choices=['health', 'predict', 'train', 'select'],
                        help='Action to perform: health, predict, train, select')
    parser.add_argument('--project', dest='project_id', type=int, default=1,
                        help='Project ID for Label Studio')
    parser.add_argument('--text', dest='text', type=str,
                        default='Apple is looking at buying U.K. startup for $1 billion.',
                        help='Text to predict entities for')
    parser.add_argument('--count', dest='count', type=int, default=10,
                        help='Number of samples to select')
    
    return parser.parse_args()

def test_health(url):
    """Test backend health endpoint"""
    response = requests.get(f"{url}/health")
    return response.json()

def test_predict(url, project_id, text):
    """Test prediction endpoint"""
    data = {
        "tasks": [{
            "id": 1,
            "data": {
                "text": text
            }
        }],
        "project": project_id,
        "label_config": """
        <View>
          <Labels name="label" toName="text">
            <Label value="PER" background="#FE9573"/>
            <Label value="ORG" background="#FBBF47"/>
            <Label value="LOC" background="#12B886"/>
            <Label value="MISC" background="#56B4E9"/>
          </Labels>
          <Text name="text" value="$text"/>
        </View>
        """
    }
    
    response = requests.post(f"{url}/predict", json=data)
    return response.json()

def test_train(url, project_id, text):
    """Test training endpoint"""
    data = {
        "event": "ANNOTATION_CREATED",
        "annotation": {
            "project": project_id,
            "result": [
                {
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": 0,
                        "end": 5,
                        "labels": ["ORG"]
                    }
                }
            ]
        },
        "task": {
            "id": 1,
            "data": {
                "text": text
            }
        }
    }
    
    response = requests.post(f"{url}/train", json=data)
    return response.json()

def test_select_samples(url, project_id, count):
    """Test sample selection endpoint"""
    data = {
        "project_id": project_id,
        "limit": count
    }
    
    response = requests.post(f"{url}/select_samples", json=data)
    return response.json()

def main():
    args = parse_args()
    
    try:
        if args.action == 'health':
            result = test_health(args.url)
            logger.info("Health check result:")
            pprint(result)
            
        elif args.action == 'predict':
            result = test_predict(args.url, args.project_id, args.text)
            logger.info("Prediction result:")
            pprint(result)
            
        elif args.action == 'train':
            result = test_train(args.url, args.project_id, args.text)
            logger.info("Training result:")
            pprint(result)
            
        elif args.action == 'select':
            result = test_select_samples(args.url, args.project_id, args.count)
            logger.info("Sample selection result:")
            pprint(result)
            
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to {args.url}. Is the server running?")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
        
    sys.exit(0)

if __name__ == "__main__":
    main() 