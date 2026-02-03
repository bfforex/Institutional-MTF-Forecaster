#!/usr/bin/env python
"""
Backtest Script

CLI script for backtesting trained models.

Usage:
    python scripts/backtest.py --model checkpoints/best_model.pt --config configs/backtest.yaml --mode confluence
"""

import argparse
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fx_hybrid.utils import load_config, setup_logger
from fx_hybrid.data import load_csv_data
from fx_hybrid.data.features_mtf import build_features_for_timeframe, select_features_for_ml, normalize_features
from fx_hybrid.indicators import build_mtf_sets
from fx_hybrid.models import HybridForexModel
from fx_hybrid.backtest import BacktestEngine, generate_full_report
from fx_hybrid.training import load_model


def main():
    parser = argparse.ArgumentParser(description='Backtest Forex Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--config', type=str, default='configs/backtest.yaml',
                       help='Path to backtest config')
    parser.add_argument('--feature-config', type=str, default='configs/features_mtf.yaml',
                       help='Path to feature config')
    parser.add_argument('--mode', type=str, choices=['base', 'smc_fvg_confluence'],
                       help='Backtest mode (overrides config)')
    parser.add_argument('--output-dir', type=str, default='backtest_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for inference')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('backtest', log_file=os.path.join(args.output_dir, 'backtest.log'))
    logger.info("Starting backtest script")
    logger.info(f"Arguments: {args}")
    
    # Load configurations
    logger.info("Loading configurations...")
    backtest_config = load_config(args.config)
    feature_config = load_config(args.feature_config)
    
    # Override mode if specified
    if args.mode:
        backtest_config['backtest']['mode'] = args.mode
        logger.info(f"Using mode: {args.mode}")
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    raw_15m = load_csv_data(args.data)
    logger.info(f"Loaded {len(raw_15m)} bars")
    
    # Build features
    logger.info("Building features...")
    features_15m = build_features_for_timeframe(
        raw_15m,
        '15min',
        fvg_config=feature_config['fvg'],
        smc_config=feature_config['smc'],
        regime_config=feature_config['regime']
    )
    
    # Select and normalize features
    features_selected = select_features_for_ml(features_15m)
    features_normalized = normalize_features(features_selected, method='zscore')
    
    # Load model
    logger.info(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=args.device)
    
    # Get model config from checkpoint metadata
    if 'metadata' in checkpoint and 'model_config' in checkpoint['metadata']:
        model_config = checkpoint['metadata']['model_config']
    else:
        # Use default config
        model_config_path = 'configs/model_base.yaml'
        model_config = load_config(model_config_path) if os.path.exists(model_config_path) else {}
        model_config = model_config.get('model', {})
    
    # Initialize model
    input_dim = features_normalized.shape[1]
    model = HybridForexModel(
        input_dim=input_dim,
        window=model_config.get('window', 256),
        conv_channels=model_config.get('conv', {}).get('channels', [32, 64]),
        kernel_size=model_config.get('conv', {}).get('kernel_size', 5),
        lstm_hidden=model_config.get('lstm', {}).get('hidden', 192),
        lstm_layers=model_config.get('lstm', {}).get('layers', 2),
        attention_dim=model_config.get('attention', {}).get('dim', 64),
        num_classes=model_config.get('num_classes', 3),
        dropout=model_config.get('lstm', {}).get('dropout', 0.2),
        film_conditioning=model_config.get('conditioning', {}).get('film_from_htf', True),
        htf_context_dim=model_config.get('conditioning', {}).get('htf_dim', 20)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Generate predictions
    logger.info("Generating predictions...")
    
    from fx_hybrid.data import create_sequences
    
    window = model_config.get('window', 256)
    feature_array = features_normalized.values
    dummy_labels = np.zeros(len(feature_array))
    
    X, _ = create_sequences(feature_array, dummy_labels, window)
    
    predictions = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(args.device)
            
            logits, _, _ = model(batch_tensor, None)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions.append(probs)
    
    predictions = np.concatenate(predictions, axis=0)
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(backtest_config['backtest'])
    
    # Align features and prices with predictions
    # Predictions start at index window-1
    aligned_features = features_15m.iloc[window-1:window-1+len(predictions)]
    aligned_prices = aligned_features['close'].values
    
    results = engine.run_backtest(
        predictions,
        aligned_features,
        aligned_prices
    )
    
    # Generate report
    logger.info("Generating report...")
    trades_df = engine.get_trades_df()
    equity_curve = engine.get_equity_curve()
    
    generate_full_report(
        results,
        trades_df,
        equity_curve,
        output_dir=args.output_dir
    )
    
    logger.info(f"\nBacktest complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
