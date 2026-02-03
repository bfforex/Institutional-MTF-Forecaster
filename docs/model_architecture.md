# Model Architecture

## Overview

The Hybrid Forex Model combines temporal convolutional networks, bidirectional LSTMs, and additive attention for multi-timeframe forex forecasting. Optional FiLM conditioning allows higher timeframes to modulate lower timeframe representations.

## Architecture Diagram

```
Input: (batch, 256, 80)
         ↓
    ┌─────────────────┐
    │  Temporal CNN   │ ← FiLM Conditioning (optional)
    │  Block 1 (32)   │    from HTF context
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │  Temporal CNN   │
    │  Block 2 (64)   │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │   Bi-LSTM       │
    │ (2 layers, 192) │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │    Additive     │
    │   Attention     │
    └─────────────────┘
         ↓
    ┌─────────────────┐
    │ Classification  │
    │   Head (128)    │
    └─────────────────┘
         ↓
    Output: (batch, 3) + confidence
```

## Components

### 1. Temporal CNN Blocks

**Purpose**: Extract local temporal patterns

```python
class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.residual = nn.Conv1d(in_channels, out_channels, 1)  # if needed
```

**Design choices**:
- **Kernel size = 5**: Captures ~75 minute patterns (5 × 15m)
- **Residual connections**: Helps gradient flow
- **BatchNorm**: Stabilizes training
- **2 blocks**: Hierarchical feature extraction

**Receptive field**:
- Block 1: 5 bars (75 minutes)
- Block 2: 9 bars (135 minutes)

### 2. FiLM Conditioning (Optional)

**Purpose**: Allow HTF context to modulate LTF features

```python
class FiLMLayer(nn.Module):
    def forward(self, x, condition):
        gamma = self.gamma_net(condition)  # Scale
        beta = self.beta_net(condition)    # Shift
        return gamma * x + beta
```

**HTF Context Vector** (20 dimensions):
- 4H/1D trend bias (1)
- 4H/1D regime (1)
- 4H/1D PD zones (2)
- 4H/1D structure ages (4)
- Key distances (4)
- Macro indicators (8)

**Benefits**:
- Macro trend awareness
- Adaptive feature extraction
- Better context understanding

**Trade-off**:
- +15% parameters
- +5% inference time
- Can be disabled if not needed

### 3. Bi-LSTM

**Purpose**: Model temporal dependencies bidirectionally

```python
self.lstm = nn.LSTM(
    input_size=64,        # From CNN output
    hidden_size=192,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.2
)
```

**Design choices**:
- **Bidirectional**: Look forward and backward in time
- **2 layers**: Balance capacity and overfitting
- **Hidden = 192**: Sufficient capacity for 80 features
- **Dropout = 0.2**: Regularization between layers

**Output**: (batch, 256, 384) → 384 = 192 × 2 (bidirectional)

### 4. Additive Attention

**Purpose**: Focus on important timesteps

```python
def additive_attention(query, values):
    # query: last LSTM hidden (batch, 384)
    # values: all LSTM outputs (batch, 256, 384)
    
    score = V(tanh(W1(query_expanded) + W2(values)))
    weights = softmax(score, dim=1)
    context = sum(weights * values)
    return context, weights
```

**Attention mechanism**:
1. Project query and values to attention space (64 dim)
2. Compute alignment scores
3. Softmax to get weights
4. Weighted sum of values

**Benefits**:
- Interpretability: Can visualize what model focuses on
- Improved performance: Learns which bars matter most
- Robustness: Less sensitive to sequence length

### 5. Classification Head

```python
self.classifier = nn.Sequential(
    nn.Linear(384, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 3)  # Up, Neutral, Down
)
```

**Output**:
- **Logits**: (batch, 3) → raw class scores
- **Probabilities**: softmax(logits) → (batch, 3)
- **Confidence**: separate head → (batch, 1)

### 6. Confidence Head

Separate output for prediction confidence:

```python
self.confidence_head = nn.Sequential(
    nn.Linear(384, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1),
    nn.Sigmoid()  # 0-1 confidence
)
```

Used for:
- Trade filtering (only trade if confidence > threshold)
- Position sizing (larger positions when confident)
- Risk management

## Model Specifications

### Default Configuration

```yaml
model:
  input_dim: 80
  window: 256          # ~64 hours at 15m
  conv_channels: [32, 64]
  kernel_size: 5
  lstm_hidden: 192
  lstm_layers: 2
  attention_dim: 64
  num_classes: 3
  dropout: 0.2
  film_conditioning: true
  htf_context_dim: 20
```

### Parameter Count

With default config:
- CNN blocks: ~40K
- LSTM: ~1.5M
- Attention: ~100K
- FiLM (if enabled): ~50K
- Classification head: ~50K
- **Total**: ~1.7-2.0M parameters

### Memory Requirements

**Training** (batch=64, AMP):
- Forward pass: ~2GB
- Backward pass: ~2GB
- Optimizer states: ~0.5GB
- **Peak**: ~5GB (fits RTX 4050)

**Inference** (batch=1):
- Forward pass: ~100MB
- **Total**: <200MB

## Training Strategy

### Loss Function

**CrossEntropy** with label smoothing:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
```

Label smoothing helps:
- Prevent overconfidence
- Better calibration
- Improved generalization

### Optimizer

**AdamW** with weight decay:

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)
```

### Learning Rate Schedule

**Cosine with warmup**:

```
LR ↑     ╱─────╲
   |    ╱       ╲
   |   ╱         ╲___
   |  ╱
   | ╱
   └────────────────→ Epoch
     │     │
   warmup  cosine decay
   (3 ep)  (27 ep)
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients during training.

### Mixed Precision (AMP)

```python
with torch.cuda.amp.autocast():
    logits, confidence, attention = model(x, htf_context)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- 2x faster training
- ~40% less memory
- Same accuracy

## Hyperparameter Tuning

### Window Size

| Window | Time Span | Use Case |
|--------|-----------|----------|
| 128 | 32 hours | Fast training, shorter patterns |
| 256 | 64 hours | **Default**, good balance |
| 512 | 128 hours | Long-term patterns, more memory |

**Recommendation**: Start with 256, reduce if memory issues.

### CNN Channels

| Config | Params | Performance |
|--------|--------|-------------|
| [16, 32] | ~0.8M | Baseline |
| [32, 64] | **~1.7M** | **Default** |
| [64, 128] | ~5M | Overkill, overfits |

**Recommendation**: [32, 64] for most cases.

### LSTM Hidden Size

| Hidden | Params | Use Case |
|--------|--------|----------|
| 96 | ~0.5M | Light model, fast |
| 192 | **~1.5M** | **Default** |
| 256 | ~2.5M | High capacity |

**Recommendation**: 192 for 80 features, scale proportionally.

### Dropout Rate

| Rate | Training | Validation |
|------|----------|------------|
| 0.1 | Faster convergence | May overfit |
| 0.2 | **Balanced** | **Recommended** |
| 0.3 | Slower | Better generalization |

**Recommendation**: 0.2, increase if overfitting.

## Model Variants

### Lightweight (Fast Inference)

```yaml
window: 128
conv_channels: [16, 32]
lstm_hidden: 96
lstm_layers: 1
film_conditioning: false
```
- Params: ~500K
- Speed: 2x faster
- Accuracy: -3% vs default

### High Capacity (Better Performance)

```yaml
window: 512
conv_channels: [64, 128]
lstm_hidden: 256
lstm_layers: 3
film_conditioning: true
```
- Params: ~8M
- Speed: 3x slower
- Accuracy: +2% vs default
- Memory: Needs >8GB VRAM

## Interpretability

### Attention Weights

```python
logits, confidence, attention_weights = model(x, htf_context)

# attention_weights: (batch, 256)
# Shows which timesteps model focuses on
```

Visualize to understand:
- Which bars influence prediction most
- If model uses recent vs historical data
- Temporal patterns learned

### Feature Importance

Use gradient-based methods:

```python
x.requires_grad = True
logits, _, _ = model(x, None)
logits.backward()

importance = x.grad.abs().mean(dim=(0, 1))  # Per feature
```

### Embedding Analysis

```python
embeddings = model.get_embeddings(x, htf_context)
# embeddings: (batch, 384)

# Use t-SNE/UMAP to visualize
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
embedded_2d = tsne.fit_transform(embeddings)
```

## Production Deployment

### ONNX Export

```bash
python scripts/export_model.py \
    --checkpoint checkpoints/best_model.pt \
    --format onnx \
    --fp16
```

**Benefits**:
- Cross-platform compatibility
- Optimized inference
- Easy integration

### TorchScript Export

```bash
python scripts/export_model.py \
    --checkpoint checkpoints/best_model.pt \
    --format torchscript
```

**Benefits**:
- No Python runtime needed
- Mobile deployment
- C++ integration

### Quantization (Optional)

For even faster inference:

```python
import torch.quantization as quantization

model_quantized = quantization.quantize_dynamic(
    model,
    {nn.LSTM, nn.Linear},
    dtype=torch.qint8
)
```

**Trade-offs**:
- 4x smaller model
- 2-3x faster inference
- ~1% accuracy drop

## Best Practices

1. ✅ **Monitor gradient norms** during training
2. ✅ **Use walk-forward validation**, not random splits
3. ✅ **Track trading metrics** (Sharpe, not just accuracy)
4. ✅ **Validate on out-of-sample** periods
5. ✅ **Regularize heavily** (dropout, weight decay, label smoothing)
6. ✅ **Use AMP** for training efficiency
7. ✅ **Save checkpoints frequently**
8. ✅ **Log attention patterns** for debugging

## Common Issues & Solutions

### Issue: Model Overfits

**Solutions**:
- Increase dropout (0.2 → 0.3)
- Add more weight decay (0.01 → 0.05)
- Reduce model capacity
- Get more data

### Issue: Training Unstable

**Solutions**:
- Reduce learning rate (3e-4 → 1e-4)
- Increase gradient clipping (1.0 → 0.5)
- Check for NaN in features
- Normalize features properly

### Issue: OOM (Out of Memory)

**Solutions**:
- Reduce batch size (64 → 32)
- Reduce window size (256 → 128)
- Reduce model size
- Enable gradient checkpointing

### Issue: Slow Inference

**Solutions**:
- Use fp16 inference
- Export to ONNX
- Batch predictions
- Use smaller model variant

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FiLM: Visual Reasoning with Feature-wise Linear Modulation](https://arxiv.org/abs/1709.07871)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

---

*For implementation, see `fx_hybrid/models/cnn_bilstm_attn.py`*
