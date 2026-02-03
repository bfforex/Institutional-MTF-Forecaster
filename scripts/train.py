#!/usr/bin/env python
"""
Training Script

CLI script for training the forex forecasting model.

Usage:
    python scripts/train.py --config configs/training.yaml --data data/EURUSD_15m.csv
"""

import argparse
import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fx_hybrid.utils import load_config, load_all_configs, setup_logger
from fx_hybrid.data import load_csv_data, create_walk_forward_splits, create_sequences
from fx_hybrid.data.features_mtf import build_features_for_timeframe, select_features_for_ml, normalize_features
from fx_hybrid.indicators import build_mtf_sets, align_to_15m
from fx_hybrid.training import ForexTrainer, generate_labels
from fx_hybrid.models import HybridForexModel, count_parameters


def main():
    parser = argparse.ArgumentParser(description='Train Forex Forecasting Model')
    parser.add_argument('--data', type=str, required=True, help='Path to 15m OHLCV CSV data')
    parser.add_argument('--config', type=str, default='configs/training.yaml', 
                       help='Path to training config')
    parser.add_argument('--model-config', type=str, default='configs/model_base.yaml',
                       help='Path to model config')
    parser.add_argument('--feature-config', type=str, default='configs/features_mtf.yaml',
                       help='Path to feature config')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('train', log_file=os.path.join(args.output_dir, 'train.log'))
    logger.info("Starting training script")
    logger.info(f"Arguments: {args}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configurations
    logger.info("Loading configurations...")
    training_config = load_config(args.config)
    model_config = load_config(args.model_config)
    feature_config = load_config(args.feature_config)
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    raw_15m = load_csv_data(args.data)
    logger.info(f"Loaded {len(raw_15m)} bars")
    
    # Build MTF features
    logger.info("Building multi-timeframe features...")
    mtf_sets = build_mtf_sets(
        raw_15m,
        frames=feature_config['timeframes']['highs']
    )
    
    # Build features for each timeframe
    logger.info("Computing indicators...")
    features_15m = build_features_for_timeframe(
        mtf_sets['15min'],
        '15min',
        fvg_config=feature_config['fvg'],
        smc_config=feature_config['smc'],
        regime_config=feature_config['regime']
    )
    
    # Select features for ML
    features_selected = select_features_for_ml(features_15m)
    features_normalized = normalize_features(features_selected, method='zscore')
    
    # Generate labels
    logger.info("Generating labels...")
    labels = generate_labels(
        features_15m,
        horizon_minutes=training_config['training']['label']['horizon_minutes'],
        band_pct=training_config['training']['label']['band_pct']
    )
    
    # Create walk-forward splits
    logger.info("Creating walk-forward splits...")
    splits = create_walk_forward_splits(
        features_normalized,
        train_months=training_config['training']['walk_forward']['train_months'],
        test_months=training_config['training']['walk_forward']['test_months'],
        embargo_bars=training_config['training']['walk_forward']['embargo_bars']
    )
    logger.info(f"Created {len(splits)} folds")
    
    # Prepare data for training
    window = model_config['model']['window']
    feature_array = features_normalized.values
    
    # Create sequences for each split
    prepared_splits = []
    for train_df, test_df in splits:
        train_features = train_df.values
        test_features = test_df.values
        
        train_labels = labels[train_df.index]
        test_labels = labels[test_df.index]
        
        # Create sequences
        X_train, y_train = create_sequences(train_features, train_labels, window)
        X_test, y_test = create_sequences(test_features, test_labels, window)
        
        # Get price series for evaluation
        train_prices = features_15m.loc[train_df.index, 'close'].values
        test_prices = features_15m.loc[test_df.index, 'close'].values
        
        prepared_splits.append((
            (X_train, y_train, None, train_prices),
            (X_test, y_test, None, test_prices)
        ))
    
    # Initialize model
    logger.info("Initializing model...")
    model = HybridForexModel(
        input_dim=feature_array.shape[1],
        window=model_config['model']['window'],
        conv_channels=model_config['model']['conv']['channels'],
        kernel_size=model_config['model']['conv']['kernel_size'],
        lstm_hidden=model_config['model']['lstm']['hidden'],
        lstm_layers=model_config['model']['lstm']['layers'],
        attention_dim=model_config['model']['attention']['dim'],
        num_classes=model_config['model']['num_classes'],
        dropout=model_config['model']['lstm']['dropout'],
        film_conditioning=model_config['model']['conditioning']['film_from_htf'],
        htf_context_dim=model_config['model']['conditioning']['htf_dim']
    )
    
    n_params = count_parameters(model)
    logger.info(f"Model has {n_params:,} trainable parameters")
    
    # Initialize trainer
    trainer_config = {
        'learning_rate': training_config['training']['learning_rate'],
        'weight_decay': training_config['training']['weight_decay'],
        'grad_clip': training_config['training']['grad_clip'],
        'label_smoothing': training_config['training']['label']['smoothing'],
        'scheduler_type': training_config['training']['scheduler']['type'],
        'warmup_epochs': training_config['training']['scheduler']['warmup_epochs'],
        'epochs_per_fold': training_config['training']['epochs_per_fold'],
        'mixed_precision': training_config['mixed_precision'],
        'num_workers': training_config['num_workers'],
        'early_stopping_patience': training_config['training']['early_stopping']['patience'],
        'early_stopping_metric': training_config['training']['early_stopping']['metric'],
        'decision_threshold': 0.55
    }
    
    trainer = ForexTrainer(model, trainer_config, device=args.device)
    
    # Train with walk-forward validation
    logger.info("Starting walk-forward training...")
    fold_metrics = trainer.walk_forward(
        prepared_splits,
        epochs_per_fold=training_config['training']['epochs_per_fold'],
        batch_size=training_config['training']['batch_size'],
        save_dir=args.output_dir
    )
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*70)
    
    for i, metrics in enumerate(fold_metrics):
        logger.info(f"\nFold {i+1}:")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}")
        logger.info(f"  Total PnL: {metrics.get('total_pnl', 0):.2f}%")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        logger.info(f"  F1 Score: {metrics.get('f1_macro', 0):.4f}")
    
    # Average metrics
    avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in fold_metrics])
    avg_sortino = np.mean([m.get('sortino_ratio', 0) for m in fold_metrics])
    avg_pnl = np.mean([m.get('total_pnl', 0) for m in fold_metrics])
    
    logger.info(f"\nAverage across folds:")
    logger.info(f"  Sharpe Ratio: {avg_sharpe:.4f}")
    logger.info(f"  Sortino Ratio: {avg_sortino:.4f}")
    logger.info(f"  Total PnL: {avg_pnl:.2f}%")
    logger.info("="*70)
    
    logger.info(f"\nCheckpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
