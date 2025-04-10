from typing import List, Tuple
import numpy as np
from wapaganda.active_learning.settings import ActiveLearningSettings

class UncertaintyEstimator:
    def __init__(self, settings: ActiveLearningSettings):
        self.settings = settings
        
    def select_uncertain_samples(
        self,
        texts: List[str],
        uncertainties: List[List[float]],
        n_samples: int | None = None
    ) -> Tuple[List[str], List[int]]:
        """
        Select samples with highest uncertainty scores.
        
        Args:
            texts: List of input texts
            uncertainties: List of uncertainty scores for each token in each text
            n_samples: Number of samples to select (defaults to samples_per_iteration)
            
        Returns:
            Tuple of selected texts and their indices
        """
        if n_samples is None:
            n_samples = self.settings.samples_per_iteration
            
        # Calculate mean uncertainty for each text
        mean_uncertainties = [np.mean(u) for u in uncertainties]
        
        # Get indices of samples with highest uncertainty
        uncertain_indices = np.argsort(mean_uncertainties)[-n_samples:]
        
        # Select corresponding texts
        selected_texts = [texts[i] for i in uncertain_indices]
        
        return selected_texts, uncertain_indices.tolist()
        
    def get_entity_boundary_uncertainty(
        self,
        predictions: List[List[str]],
        uncertainties: List[List[float]]
    ) -> List[float]:
        """
        Calculate uncertainty scores considering entity boundaries.
        
        Args:
            predictions: Predicted entity labels for each token
            uncertainties: Uncertainty scores for each token
            
        Returns:
            List of boundary-aware uncertainty scores
        """
        boundary_uncertainties = []
        
        for text_preds, text_uncs in zip(predictions, uncertainties):
            # Higher weight for tokens at entity boundaries
            boundary_weight = 1.5
            weighted_uncs = []
            
            for i, (pred, unc) in enumerate(zip(text_preds, text_uncs)):
                is_boundary = (
                    pred.startswith("B-") or
                    (i > 0 and text_preds[i-1].startswith("I-") and not pred.startswith("I-")) or
                    (i < len(text_preds)-1 and not pred.startswith("I-") and text_preds[i+1].startswith("B-"))
                )
                
                weighted_uncs.append(unc * (boundary_weight if is_boundary else 1.0))
                
            boundary_uncertainties.append(np.mean(weighted_uncs))
            
        return boundary_uncertainties

# settings = ActiveLearningSettings() 