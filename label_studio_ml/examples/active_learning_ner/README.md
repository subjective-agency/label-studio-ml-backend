# Active Learning NER with Label Studio ML Backend

This backend implements an active learning pipeline for Named Entity Recognition (NER) tasks, integrated with Label Studio. 
It enhances the standard NER capabilities with uncertainty-based sampling to efficiently build large NER datasets 
starting with a minimal set of labeled examples.

## Features

- **Active Learning** - Intelligently selects the most uncertain samples for annotation
- **Entity Boundary Detection** - Focuses on entity boundaries to improve annotation efficiency
- **MLflow Integration** - Tracks experiments and model performance metrics
- **Hugging Face Integration** - Uses transformer-based models for state-of-the-art NER
- **Docker Setup** - Easy deployment with Docker Compose

## Setup

### Prerequisites

- Docker and Docker Compose
- Label Studio instance (accessible from the backend container)
- API key for Label Studio

### Quick Start

1. Clone the Label Studio ML Backend repository:
   ```bash
   git clone https://github.com/HumanSignal/label-studio-ml-backend.git
   cd label-studio-ml-backend
   ```

2. Set your Label Studio API key in an environment variable:
   ```bash
   export LABEL_STUDIO_API_KEY=your_api_key_here
   ```

3. Build and start the Active Learning NER backend:
   ```bash
   cd label_studio_ml/examples/active_learning_ner
   docker-compose up -d
   ```

4. Create necessary directories:
   ```bash
   mkdir -p data/server/models data/.cache data/mlruns
   ```

5. The backend will be available at http://localhost:9090

### Connecting to Label Studio

1. Create a new project in Label Studio
2. Add a Named Entity Recognition labeling interface:
   ```xml
   <View>
     <Labels name="label" toName="text">
       <Label value="PER" background="#FE9573"/>
       <Label value="ORG" background="#FBBF47"/>
       <Label value="LOC" background="#12B886"/>
       <Label value="MISC" background="#56B4E9"/>
     </Labels>
     <Text name="text" value="$text"/>
   </View>
   ```
3. Connect the ML backend to your project:
   - In your project, go to Settings > Machine Learning
   - Add a new model with the URL: http://localhost:9090
   - Click "Add Model"

## Configuration

The backend can be configured with the following environment variables in the docker-compose.yml file:

| Variable | Description | Default |
|----------|-------------|---------|
| LABEL_STUDIO_HOST | URL of your Label Studio instance | http://localhost:8080 |
| LABEL_STUDIO_API_KEY | API key for Label Studio | (required) |
| BASELINE_MODEL_NAME | Base Hugging Face model to use | dslim/bert-base-NER |
| FINETUNED_MODEL_NAME | Directory name for saving fine-tuned models | finetuned_model |
| START_TRAINING_EACH_N_UPDATES | Number of annotations before starting training | 10 |
| LEARNING_RATE | Learning rate for model training | 2e-5 |
| NUM_TRAIN_EPOCHS | Number of training epochs | 3 |
| WEIGHT_DECAY | Weight decay for optimizer | 0.01 |
| MAX_SEQUENCE_LENGTH | Maximum sequence length for tokenization | 128 |
| UNCERTAINTY_THRESHOLD | Threshold for selecting uncertain samples | 0.8 |
| SAMPLES_PER_ITERATION | Number of samples to select in each active learning iteration | 500 |
| MLFLOW_TRACKING_URI | URI for MLflow tracking | sqlite:///mlruns.db |
| LOG_LEVEL | Logging level | INFO |

## Active Learning Workflow

1. **Initial Data Annotation**:
   - Upload a small batch of texts (around 200 samples)
   - Manually annotate them to create an initial training set

2. **Initial Model Training**:
   - After annotating the initial set, the model will train automatically when reaching the number of annotations specified in `START_TRAINING_EACH_N_UPDATES`
   - You can also start training manually from the ML backend settings in Label Studio

3. **Upload Unlabeled Data**:
   - Upload a larger batch of unlabeled texts (thousands of samples)

4. **Active Learning Cycle**:
   - The model will predict entities in unlabeled texts
   - It will calculate uncertainty scores for each prediction
   - The most uncertain samples will be prioritized for annotation
   - Each time you annotate a batch of samples, the model retrains automatically
   - Performance metrics are tracked in MLflow

5. **Model Improvement**:
   - As more annotations are added, the model becomes more accurate
   - The active learning approach ensures efficient use of annotation resources
   - You can track progress through the MLflow dashboard

## Accessing MLflow Dashboard

The MLflow dashboard is available through the container at http://localhost:5000 if you expose the port in docker-compose.yml:

```yaml
services:
  active_learning_ner:
    # ... other settings
    ports:
      - "9090:9090"
      - "5000:5000"  # Add this line
```

You'll need to run MLflow UI inside the container:

```bash
docker exec -it active_learning_ner mlflow ui --host 0.0.0.0
```

## Implementation Details

The backend implements:

1. **Uncertainty Estimation** - Calculates token-level uncertainty using probability scores from the model
2. **Entity Boundary Focus** - Weighs uncertainty scores higher at entity boundaries
3. **Sample Selection** - Selects samples with the highest uncertainty scores for annotation
4. **Model Training** - Fine-tunes a Hugging Face transformer model with new annotations
5. **Performance Tracking** - Logs metrics like F1 score, precision, and recall to MLflow

## Advanced Usage

### Custom Entity Types

To use custom entity types, modify the labeling configuration in Label Studio. The backend will automatically detect and use these entity types during training.

### Pre-annotation

The backend supports pre-annotation of unlabeled texts. The model will predict entity labels and display them in the Label Studio interface for you to verify or correct.

### Batch Selection API

You can use the backend API to manually select uncertain samples:

```bash
curl -X POST http://localhost:9090/select_samples \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "limit": 100}'
```

This endpoint returns the IDs of the most uncertain samples for annotation.

## Troubleshooting

- **Model not training**: Check if you have reached the required number of annotations (`START_TRAINING_EACH_N_UPDATES`)
- **Low performance**: Try increasing `NUM_TRAIN_EPOCHS` or decreasing `LEARNING_RATE`
- **Memory issues**: Reduce `MAX_SEQUENCE_LENGTH` or use a smaller base model
- **Connection errors**: Ensure Label Studio is accessible from the Docker container network

## References

- [Label Studio ML Backend](https://github.com/HumanSignal/label-studio-ml-backend)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Active Learning Literature](https://arxiv.org/abs/2009.00236)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) 