"""
Continuous Learning Module

Tracks prediction errors and adjusts training for difficult patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class MistakeTracker:
    """
    Track prediction errors and adjust sample weighting.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Parameters
        ----------
        window_size : int
            Rolling window for tracking recent mistakes
        """
        self.window_size = window_size
        self.predictions = []
        self.actuals = []
        self.features = []
        self.losses = []
        self.timestamps = []
        
        self.pattern_errors = defaultdict(list)
        self.feature_importance = {}
    
    def log_prediction(
        self,
        pred: np.ndarray,
        actual: int,
        features: np.ndarray,
        loss: float,
        timestamp: int = None
    ):
        """
        Log a prediction for later analysis.
        
        Parameters
        ----------
        pred : np.ndarray
            Model prediction (probabilities)
        actual : int
            True label
        features : np.ndarray
            Input features
        loss : float
            Loss value for this prediction
        timestamp : int, optional
            Timestamp or index
        """
        self.predictions.append(pred)
        self.actuals.append(actual)
        self.features.append(features)
        self.losses.append(loss)
        self.timestamps.append(timestamp if timestamp is not None else len(self.predictions))
        
        # Keep only recent history
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.actuals.pop(0)
            self.features.pop(0)
            self.losses.pop(0)
            self.timestamps.pop(0)
    
    def get_sample_weights(
        self,
        current_features: np.ndarray,
        method: str = 'loss_weighted'
    ) -> np.ndarray:
        """
        Get adjusted sample weights based on historical mistakes.
        
        Parameters
        ----------
        current_features : np.ndarray
            Current batch features
        method : str
            Weighting method: 'loss_weighted' or 'pattern_match'
            
        Returns
        -------
        np.ndarray
            Sample weights
        """
        n_samples = len(current_features)
        weights = np.ones(n_samples)
        
        if len(self.losses) == 0:
            return weights
        
        if method == 'loss_weighted':
            # Weight samples inversely proportional to historical performance
            # on similar patterns
            avg_loss = np.mean(self.losses)
            
            for i in range(n_samples):
                # Find similar patterns in history
                similarities = self._compute_similarities(current_features[i])
                
                if len(similarities) > 0:
                    # Weight by average loss of similar patterns
                    similar_losses = [self.losses[j] for j in similarities[:10]]  # Top 10
                    avg_similar_loss = np.mean(similar_losses)
                    
                    # Increase weight for difficult patterns
                    weights[i] = max(1.0, avg_similar_loss / (avg_loss + 1e-6))
        
        return weights
    
    def _compute_similarities(
        self,
        feature: np.ndarray,
        top_k: int = 10
    ) -> List[int]:
        """
        Find similar historical patterns.
        
        Parameters
        ----------
        feature : np.ndarray
            Query feature vector
        top_k : int
            Number of similar patterns to return
            
        Returns
        -------
        List[int]
            Indices of similar patterns
        """
        if len(self.features) == 0:
            return []
        
        # Compute cosine similarity
        similarities = []
        for i, hist_feature in enumerate(self.features):
            # Flatten if needed
            feat_flat = feature.flatten()
            hist_flat = hist_feature.flatten()
            
            # Ensure same length
            min_len = min(len(feat_flat), len(hist_flat))
            feat_flat = feat_flat[:min_len]
            hist_flat = hist_flat[:min_len]
            
            # Cosine similarity
            norm_prod = (np.linalg.norm(feat_flat) * np.linalg.norm(hist_flat))
            if norm_prod > 0:
                sim = np.dot(feat_flat, hist_flat) / norm_prod
                similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, sim in similarities[:top_k]]
    
    def analyze_errors(self) -> Dict:
        """
        Analyze error patterns.
        
        Returns
        -------
        Dict
            Error analysis report
        """
        if len(self.predictions) == 0:
            return {}
        
        predictions_array = np.array(self.predictions)
        actuals_array = np.array(self.actuals)
        losses_array = np.array(self.losses)
        
        # Class-wise errors
        class_errors = {}
        for cls in [0, 1, 2]:
            mask = actuals_array == cls
            if mask.sum() > 0:
                class_errors[f'class_{cls}_avg_loss'] = losses_array[mask].mean()
                class_errors[f'class_{cls}_count'] = mask.sum()
        
        # Overall statistics
        report = {
            'total_samples': len(self.predictions),
            'avg_loss': losses_array.mean(),
            'max_loss': losses_array.max(),
            'min_loss': losses_array.min(),
            **class_errors
        }
        
        return report
    
    def get_calibration_adjustments(self) -> Dict[int, float]:
        """
        Compute calibration adjustments for each class.
        
        Returns
        -------
        Dict[int, float]
            Calibration adjustments per class
        """
        if len(self.predictions) == 0:
            return {0: 1.0, 1: 1.0, 2: 1.0}
        
        predictions_array = np.array(self.predictions)
        actuals_array = np.array(self.actuals)
        
        adjustments = {}
        
        for cls in [0, 1, 2]:
            # Get predictions where model predicted this class
            pred_cls = predictions_array.argmax(axis=1)
            pred_this_cls = pred_cls == cls
            
            if pred_this_cls.sum() > 0:
                # Calculate accuracy for this class
                correct = (actuals_array[pred_this_cls] == cls).sum()
                total = pred_this_cls.sum()
                accuracy = correct / total
                
                # Adjustment: reduce confidence if accuracy is low
                adjustments[cls] = accuracy
            else:
                adjustments[cls] = 1.0
        
        return adjustments


def apply_mistake_tracking(
    model_output: np.ndarray,
    tracker: MistakeTracker
) -> np.ndarray:
    """
    Apply calibration adjustments to model output.
    
    Parameters
    ----------
    model_output : np.ndarray
        Raw model probabilities
    tracker : MistakeTracker
        Mistake tracker with calibration info
        
    Returns
    -------
    np.ndarray
        Calibrated probabilities
    """
    adjustments = tracker.get_calibration_adjustments()
    
    calibrated = model_output.copy()
    
    for cls in [0, 1, 2]:
        calibrated[:, cls] *= adjustments[cls]
    
    # Re-normalize
    row_sums = calibrated.sum(axis=1, keepdims=True)
    calibrated = calibrated / (row_sums + 1e-10)
    
    return calibrated
