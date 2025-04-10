# Step-by-Step Guide: Building a 30K NER Dataset Using Active Learning

This guide outlines a comprehensive approach to efficiently build a large Named Entity Recognition (NER) dataset (approximately 30,000 records) starting with only ~200 manually labeled samples. The methodology leverages active learning techniques to minimize annotation costs while maximizing model performance.

## Prerequisites

- ~200 manually annotated NER samples (sentences with entity labels)
- Unlabeled text corpus from your target domain (at least 50K-100K sentences)
- Access to computing resources for model training
- Annotation tools that support active learning workflows (e.g., Prodigy, Doccano, Label Studio)
- Basic understanding of NLP and NER concepts

## Phase 1: Initial Setup and Foundation (Weeks 1-2)

### Step 1: Prepare Your Initial Dataset
1. Ensure your 200 manually labeled samples:
   - Cover diverse entity types in your domain
   - Represent various text structures and patterns
   - Are correctly labeled with consistent annotation guidelines
   - Are split into training (160 samples) and validation (40 samples) sets

### Step 2: Select a Base Model
1. Choose an appropriate foundation model:
   - For general domains: BERT, RoBERTa, or DeBERTa base models
   - For specialized domains: BioClinicalBERT (medical), FinBERT (finance), etc.
   - Consider computational constraints when selecting model size

### Step 3: Implement the Active Learning Pipeline
1. Set up a modular pipeline with these components:
   - Model training module
   - Uncertainty estimation module
   - Sample selection module
   - Annotation interface connector
   - Model evaluation module
2. Integrate with annotation tools:
   - Prodigy's `ner.teach` workflow
   - Doccano with custom export/import scripts
   - Label Studio with ML backend integration

## Phase 2: Initial Model Training (Week 3)

### Step 4: Train the First Model
1. Fine-tune your base model on the initial 200 samples:
   - Use appropriate hyperparameters (learning rate: 2e-5 to 5e-5)
   - Apply early stopping based on validation performance
   - Save model checkpoints for later evaluation
2. Evaluate model performance:
   - Entity-level precision, recall, F1 score
   - Document baseline metrics for comparison

## Phase 3: First Active Learning Cycle (Weeks 4-6)

### Step 5: Uncertainty-Based Sample Selection
1. Apply the trained model to 5,000-10,000 unlabeled samples
2. Calculate uncertainty scores using:
   - Token-level entropy: `H(y|x) = -∑ P(y|x) log P(y|x)`
   - Sequence-level uncertainty: average token uncertainties or sequence margin
   - Entity boundary uncertainty: focusing on tokens with ambiguous entity boundaries
3. Select 500-1,000 most uncertain samples for annotation

### Step 6: First Batch Annotation
1. Present selected samples to annotators through your chosen tool
2. Annotate following consistent guidelines
3. Implement quality control:
   - Cross-check 10% of annotations
   - Calculate inter-annotator agreement (aim for Cohen's kappa > 0.8)
4. Add new annotations to the training set

### Step 7: Model Retraining and Evaluation
1. Retrain the model on the expanded dataset (~1,000-1,200 samples)
2. Evaluate on the validation set
3. Document improvements in F1 score and other metrics
4. Analyze error patterns to refine future selection strategies

## Phase 4: Hybrid Selection Strategy (Weeks 7-10)

### Step 8: Implement Diversity-Based Sampling
1. Extract features from unlabeled data:
   - Contextual embeddings from the current model
   - Linguistic features (POS tags, dependency structure)
   - Domain-specific features if available
2. Apply clustering algorithms:
   - K-means clustering (k=10-20)
   - Density-based clustering (DBSCAN)
3. Define a hybrid selection function:
   - α × uncertainty_score + (1-α) × diversity_score (start with α=0.7)
   - Select samples across different clusters, prioritizing uncertain ones

### Step 9: Second Batch Annotation (Larger Scale)
1. Select 2,000-3,000 samples using the hybrid strategy
2. Distribute annotation tasks to multiple annotators if possible
3. Implement batch processing to maintain annotation quality
4. Add new annotations to the training set (total ~3,000-4,000 samples)

### Step 10: Enhanced Model Training
1. Retrain the model with:
   - Appropriate learning rate decay
   - Balanced sampling to handle entity class imbalance
   - Data augmentation techniques (synonym replacement, entity swapping)
2. Evaluate performance improvements
3. Implement entity-specific metrics to identify weaknesses

## Phase 5: Scaling Up (Weeks 11-16)

### Step 11: Entity-Aware Selection
1. Implement subsequence selection methods:
   - Focus on entity-rich portions of text
   - Prioritize contexts with potential rare entity types
   - Implement EASAL or similar entity-aware approaches
2. Select 5,000-8,000 samples for the next annotation batch

### Step 12: Large-Scale Annotation
1. Scale annotation process:
   - Consider utilizing managed annotation services
   - Implement stronger quality assurance processes
   - Use pre-annotation with model predictions to speed up the process
2. Add new annotations to reach ~10,000 labeled samples

### Step 13: Advanced Model Training
1. Train larger models or ensembles:
   - Experiment with model architectures (Transformer-CRF hybrids)
   - Implement query-by-committee approaches with model ensembles
   - Fine-tune hyperparameters more extensively
2. Evaluate performance improvements
3. Estimate the annotation budget needed to reach target dataset size

## Phase 6: Final Scaling and Refinement (Weeks 17-24)

### Step 14: Token Re-weighting and Advanced Selection
1. Implement token re-weighting strategies:
   - Assign higher weights to rare entity types
   - Use dynamic smoothing weights based on token frequencies
   - Focus on boundary tokens with high uncertainty
2. Select final 15,000-20,000 samples using optimized strategies

### Step 15: Final Annotation Batches
1. Implement streamlined annotation workflows:
   - Use model pre-annotations with human verification
   - Focus annotator attention on uncertain predictions
   - Batch similar examples to improve annotation efficiency
2. Process annotations in multiple batches (5,000 samples each)
3. Track annotation speed and quality metrics

### Step 16: Final Dataset Compilation
1. Combine all annotated data (~30,000 samples)
2. Clean and standardize the dataset:
   - Fix inconsistent annotations
   - Remove duplicates
   - Ensure format compatibility
3. Create official train/validation/test splits:
   - 80% training (24,000 samples)
   - 10% validation (3,000 samples)
   - 10% test (3,000 samples)

## Phase 7: Evaluation and Documentation (Weeks 25-26)

### Step 17: Comprehensive Evaluation
1. Train final models on the complete dataset
2. Evaluate across multiple dimensions:
   - Overall NER performance (precision, recall, F1)
   - Entity-specific metrics
   - Performance on challenging subsets
   - Comparison with initial performance

### Step 18: Dataset Documentation
1. Document the dataset creation process:
   - Annotation guidelines
   - Active learning strategy
   - Dataset statistics and characteristics
   - Known limitations
2. Create dataset cards following best practices

## Key Performance Indicators Throughout the Process

| Phase | Dataset Size | Expected F1 Score | Annotation Efficiency |
|-------|-------------|------------------|----------------------|
| Initial | 200 | 50-60% | 100% manual |
| Phase 3 | ~1,200 | 65-75% | 60-70% manual effort |
| Phase 4 | ~4,000 | 75-82% | 40-50% manual effort |
| Phase 5 | ~10,000 | 82-88% | 30-40% manual effort |
| Phase 6 | ~30,000 | 88-95% | 20-30% manual effort |

## Best Practices for Success

1. **Maintain consistent annotation guidelines** throughout the process
2. **Regularly evaluate model performance** to adjust active learning strategies
3. **Balance between uncertainty and diversity** in sample selection
4. **Focus on problematic entity types** that show lower performance
5. **Use pre-annotation** for later batches to speed up the process
6. **Implement strong quality control** to ensure annotation consistency
7. **Document each iteration** thoroughly for reproducibility
8. **Consider computational efficiency** when scaling up
9. **Adapt the strategy** based on performance patterns and domain specifics
10. **Regularly refresh annotator training** to maintain high-quality annotations

## Tools and Resources

1. **Annotation Platforms:**
   - [Prodigy](https://prodi.gy/) (commercial, built-in active learning support)
   - [Doccano](https://github.com/doccano/doccano) (open-source)
   - [Label Studio](https://labelstud.io/) (open-source with enterprise version)

2. **Model Implementation:**
   - [Hugging Face Transformers](https://github.com/huggingface/transformers)
   - [spaCy](https://spacy.io/) (with prodigy integration)
   - [NERDA](https://github.com/ebanalyse/NERDA) (specialized NER tools)

3. **Active Learning Libraries:**
   - [ModAL](https://github.com/modAL-python/modAL)
   - [ALiPy](https://github.com/NUAA-AL/ALiPy)
   - Custom implementations based on PyTorch/TensorFlow

## Conclusion

Following this step-by-step guide should enable you to efficiently build a high-quality NER dataset of approximately 30,000 annotated samples starting from just 200 manually labeled examples. By leveraging active learning techniques, you can significantly reduce annotation costs while maximizing model performance. The approach balances uncertainty-based and diversity-based sampling strategies, incorporates entity-aware methods, and scales efficiently to reach the target dataset size.