"""
Training Module

Main training loop with AMP, walk-forward validation, and comprehensive metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import os

from .metrics import compute_all_metrics
from .continuous_learning import MistakeTracker


class ForexDataset(Dataset):
    """PyTorch Dataset for forex sequences."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        htf_context: Optional[np.ndarray] = None
    ):
        """
        Parameters
        ----------
        features : np.ndarray
            Feature sequences of shape (n_samples, window, features)
        labels : np.ndarray
            Labels of shape (n_samples,)
        htf_context : np.ndarray, optional
            HTF context vectors of shape (n_samples, htf_dim)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
        if htf_context is not None:
            self.htf_context = torch.FloatTensor(htf_context)
        else:
            self.htf_context = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.htf_context is not None:
            return self.features[idx], self.labels[idx], self.htf_context[idx]
        else:
            return self.features[idx], self.labels[idx], None


class ForexTrainer:
    """
    Main trainer class with AMP and walk-forward validation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Parameters
        ----------
        model : nn.Module
            Model to train
        config : Dict
            Training configuration
        device : str
            Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.05)
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler(config)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.get('mixed_precision', True) else None
        
        # Gradient clipping
        self.grad_clip = config.get('grad_clip', 1.0)
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Mistake tracker
        self.mistake_tracker = MistakeTracker()
        
        # Best model tracking
        self.best_metric = float('-inf')
        self.best_epoch = 0
    
    def _create_scheduler(self, config):
        """Create learning rate scheduler."""
        scheduler_type = config.get('scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            warmup_epochs = config.get('warmup_epochs', 3)
            total_epochs = config.get('epochs_per_fold', 30)
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        
        else:
            return None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        epoch : int
            Current epoch number
            
        Returns
        -------
        float
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            features, labels, htf_context = batch
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            if htf_context is not None and htf_context[0] is not None:
                htf_context = htf_context.to(self.device)
            else:
                htf_context = None
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, confidence, _ = self.model(features, htf_context)
                    loss = self.criterion(logits, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, confidence, _ = self.model(features, htf_context)
                loss = self.criterion(logits, labels)
                loss.backward()
                
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader,
        prices: np.ndarray
    ) -> Dict[str, float]:
        """
        Validate model and compute all metrics.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
        prices : np.ndarray
            Price series for computing trading metrics
            
        Returns
        -------
        Dict[str, float]
            Dictionary of all metrics
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features, labels, htf_context = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                if htf_context is not None and htf_context[0] is not None:
                    htf_context = htf_context.to(self.device)
                else:
                    htf_context = None
                
                logits, confidence, _ = self.model(features, htf_context)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                n_batches += 1
                
                # Collect predictions
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_predictions.append(probs)
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)
        
        # Concatenate all predictions
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # Compute metrics
        metrics = compute_all_metrics(
            predictions,
            labels,
            prices[:len(predictions)+1],
            threshold=self.config.get('decision_threshold', 0.55)
        )
        
        metrics['val_loss'] = avg_loss
        self.val_metrics.append(metrics)
        
        return metrics
    
    def walk_forward(
        self,
        splits: List[Tuple],
        epochs_per_fold: int = 30,
        batch_size: int = 64,
        save_dir: str = 'checkpoints'
    ) -> List[Dict]:
        """
        Perform walk-forward validation.
        
        Parameters
        ----------
        splits : List[Tuple]
            List of (train_data, val_data) tuples
        epochs_per_fold : int
            Number of epochs per fold
        batch_size : int
            Batch size
        save_dir : str
            Directory to save checkpoints
            
        Returns
        -------
        List[Dict]
            List of validation metrics for each fold
        """
        os.makedirs(save_dir, exist_ok=True)
        
        all_fold_metrics = []
        
        for fold_idx, (train_data, val_data) in enumerate(splits):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{len(splits)}")
            print(f"{'='*50}")
            
            # Create data loaders
            train_dataset = ForexDataset(*train_data)
            val_dataset = ForexDataset(*val_data)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.config.get('num_workers', 4)
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.get('num_workers', 4)
            )
            
            # Reset tracking for this fold
            self.best_metric = float('-inf')
            self.best_epoch = 0
            patience_counter = 0
            patience = self.config.get('early_stopping_patience', 5)
            
            # Train for epochs
            for epoch in range(1, epochs_per_fold + 1):
                train_loss = self.train_epoch(train_loader, epoch)
                
                # Get validation prices from val_data tuple
                # Assuming val_data structure: (features, labels, htf_context, prices)
                if len(val_data) >= 4:
                    val_prices = val_data[3]
                else:
                    # Placeholder if prices not provided
                    val_prices = np.arange(len(val_data[1]) + 1) * 1.0
                
                val_metrics = self.validate(val_loader, val_prices)
                
                # Print metrics
                print(f"\nEpoch {epoch}/{epochs_per_fold}")
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}")
                print(f"Sharpe: {val_metrics['sharpe_ratio']:.4f}, Sortino: {val_metrics['sortino_ratio']:.4f}")
                
                # Check for improvement
                metric_to_monitor = self.config.get('early_stopping_metric', 'sharpe_ratio')
                current_metric = val_metrics[metric_to_monitor]
                
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    checkpoint_path = os.path.join(save_dir, f'best_model_fold{fold_idx}.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'metrics': val_metrics
                    }, checkpoint_path)
                    print(f"✓ New best model saved (Fold {fold_idx}, Epoch {epoch})")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Load best model for this fold
            checkpoint_path = os.path.join(save_dir, f'best_model_fold{fold_idx}.pt')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                fold_metrics = checkpoint['metrics']
            else:
                fold_metrics = val_metrics
            
            all_fold_metrics.append(fold_metrics)
            
            print(f"\nFold {fold_idx + 1} completed. Best epoch: {self.best_epoch}")
            print(f"Best {metric_to_monitor}: {self.best_metric:.4f}")
        
        return all_fold_metrics


def save_model(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict] = None
):
    """Save model with metadata."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(
    model: nn.Module,
    path: str,
    device: str = 'cuda'
) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model
