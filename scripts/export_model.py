#!/usr/bin/env python
"""
Model Export Script

Export trained models to ONNX or TorchScript format.

Usage:
    python scripts/export_model.py --checkpoint checkpoints/best_model.pt --format onnx --fp16
"""

import argparse
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fx_hybrid.utils import load_config, setup_logger
from fx_hybrid.models import HybridForexModel


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple,
    opset_version: int = 11,
    fp16: bool = False
):
    """
    Export model to ONNX format.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to export
    output_path : str
        Path to save ONNX model
    input_shape : tuple
        Input shape (batch, window, features)
    opset_version : int
        ONNX opset version
    fp16 : bool
        Use FP16 precision
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    if fp16:
        model = model.half()
        dummy_input = dummy_input.half()
    
    # Export
    torch.onnx.export(
        model,
        (dummy_input, None),  # Input tuple
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input', 'htf_context'],
        output_names=['logits', 'confidence', 'attention_weights'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'htf_context': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'confidence': {0: 'batch_size'},
            'attention_weights': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {output_path}")


def export_to_torchscript(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple,
    fp16: bool = False
):
    """
    Export model to TorchScript format.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to export
    output_path : str
        Path to save TorchScript model
    input_shape : tuple
        Input shape
    fp16 : bool
        Use FP16 precision
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    if fp16:
        model = model.half()
        dummy_input = dummy_input.half()
    
    # Trace model
    traced_model = torch.jit.trace(model, (dummy_input, None))
    
    # Save
    traced_model.save(output_path)
    
    print(f"Model exported to TorchScript: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export Forex Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript'],
                       default='onnx', help='Export format')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--fp16', action='store_true',
                       help='Export with FP16 precision')
    parser.add_argument('--model-config', type=str, default='configs/model_base.yaml',
                       help='Path to model config')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('export')
    logger.info("Starting model export")
    logger.info(f"Arguments: {args}")
    
    # Load model config
    if os.path.exists(args.model_config):
        model_config = load_config(args.model_config)
        model_config = model_config.get('model', {})
    else:
        logger.warning(f"Model config not found: {args.model_config}, using defaults")
        model_config = {}
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Extract model config from checkpoint if available
    if 'metadata' in checkpoint and 'model_config' in checkpoint['metadata']:
        model_config = checkpoint['metadata']['model_config']
    
    # Initialize model
    input_dim = model_config.get('features', 80)
    window = model_config.get('window', 256)
    
    model = HybridForexModel(
        input_dim=input_dim,
        window=window,
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Determine output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        ext = '.onnx' if args.format == 'onnx' else '.pt'
        args.output = f"{base_name}_exported{ext}"
    
    # Input shape for export
    input_shape = (1, window, input_dim)
    
    # Export
    logger.info(f"Exporting to {args.format.upper()}...")
    
    if args.format == 'onnx':
        export_to_onnx(
            model,
            args.output,
            input_shape,
            opset_version=11,
            fp16=args.fp16
        )
    else:  # torchscript
        export_to_torchscript(
            model,
            args.output,
            input_shape,
            fp16=args.fp16
        )
    
    logger.info(f"Export complete: {args.output}")
    
    # Print file size
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    logger.info(f"File size: {file_size:.2f} MB")


if __name__ == '__main__':
    main()
