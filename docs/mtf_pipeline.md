# Multi-Timeframe (MTF) Pipeline

## Overview

The MTF pipeline synchronizes features across multiple timeframes (15m, 1H, 4H, 1D) while maintaining strict anti-leakage guarantees. This enables the model to leverage both short-term patterns and longer-term context.

## Architecture

```
Raw 15m Data
     ↓
┌────────────────────────────────┐
│  Resample to HTF (1H, 4H, 1D)  │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│   Compute Indicators Per TF    │
│  (FVG, SMC, Regime for each)   │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│  Align HTF → 15m Timeline      │
│  (Forward-fill with shift)     │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│    Merge All Features → ML     │
│  (15m + 1H + 4H + 1D features) │
└────────────────────────────────┘
```

## Resampling Strategy

### Calendar-Aware Resampling

Uses pandas `resample()` with proper frequency strings:
- `15min` → base timeframe (no resampling)
- `1H` → 1-hour bars
- `4H` → 4-hour bars
- `1D` → daily bars

**OHLCV Aggregation**:
- Open: First value in period
- High: Maximum in period
- Low: Minimum in period
- Close: Last value in period
- Volume: Sum of period

```python
def resample_ohlcv(df, target_freq):
    resampled = pd.DataFrame()
    resampled['open'] = df['open'].resample(target_freq).first()
    resampled['high'] = df['high'].resample(target_freq).max()
    resampled['low'] = df['low'].resample(target_freq).min()
    resampled['close'] = df['close'].resample(target_freq).last()
    resampled['volume'] = df['volume'].resample(target_freq).sum()
    return resampled
```

## Alignment Methods

### Broadcast (Forward-Fill)

The recommended method for HTF → LTF alignment:

```python
def align_to_15m(htf_df, master_15m_index, anti_leakage=True):
    # Reindex to 15m timeline
    aligned = htf_df.reindex(master_15m_index, method='ffill')
    
    if anti_leakage:
        # Shift by 1 to only use completed HTF bars
        aligned = aligned.shift(1)
    
    aligned = aligned.fillna(method='bfill').fillna(0)
    return aligned
```

**Key points**:
1. HTF values are forward-filled to all 15m bars within the HTF period
2. **Anti-leakage**: Shift by 1 ensures we never use incomplete HTF bars
3. Initial NaNs (before first complete HTF bar) are back-filled or set to 0

### Example: 1H Alignment

```
1H bars:     [  A  ][  B  ][  C  ][  D  ]
15m bars:    ▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢...
Aligned:     0000AAAABBBBCCCCDDDD...
             ↑   ↑
           shift  A available here
```

At 15m bar 5, we have access to completed 1H bar A (bars 0-3 were in 1H bar A).

## Anti-Leakage Guarantees

### The Problem

Without proper handling, HTF features can leak future information:

```
❌ WRONG: Using current HTF bar values
15m:  t0  t1  t2  t3 | t4  t5  t6  t7
1H:   [    Bar N    ] [    Bar N+1   ]
      
At t2, using Bar N is OK
At t4, using Bar N+1 is LEAKING (bar not complete)
```

### The Solution

Always use completed HTF bars only:

```
✓ CORRECT: Shift by 1, use completed bars
15m:  t0  t1  t2  t3 | t4  t5  t6  t7 | t8
1H:   [    Bar N    ] [    Bar N+1   ]
Used:  0   0   0   0    N   N   N   N   N+1
                        ↑
              Bar N completed, now available
```

### Validation Tests

Run anti-leakage tests:

```python
from fx_hybrid.indicators.mtf import ensure_no_leakage

# Check alignment
is_valid = ensure_no_leakage(df_15m, df_htf_aligned, 'HTF_name')
```

Tests verify:
1. HTF values only change at proper boundaries
2. No forward-looking information
3. Reasonable NaN handling

## Feature Merging

### Feature Count Budget

To stay within GPU memory (6GB):

| Timeframe | Features | Notes |
|-----------|----------|-------|
| 15m | 40 | Full feature set |
| 1H | 20 | Selected features |
| 4H | 10 | Key features only |
| 1D | 10 | Macro features |
| **Total** | **80** | Target ≤ 96 |

### Feature Selection Per TF

**15m (Low Timeframe)**:
- All FVG features
- All SMC features
- Regime
- Price/volume

**1H (Mid Timeframe)**:
- Key FVG: distances, flags
- Key SMC: trend bias, BOS/CHOCH ages
- PD zones
- Regime

**4H/1D (High Timeframe)**:
- Trend bias
- Major structure flags
- Regime
- Key distances

### Merging Process

```python
def merge_features(fts_15m, fts_1H, fts_4H, fts_1D):
    # Add prefixes
    fts_15m_renamed = fts_15m.add_prefix('15m_')
    fts_1H_renamed = fts_1H.add_prefix('1h_')
    fts_4H_renamed = fts_4H.add_prefix('4h_')
    fts_1D_renamed = fts_1D.add_prefix('1d_')
    
    # Concatenate
    merged = pd.concat([
        fts_15m_renamed,
        fts_1H_renamed,
        fts_4H_renamed,
        fts_1D_renamed
    ], axis=1)
    
    return merged.values
```

## Complete Pipeline Example

```python
from fx_hybrid.indicators import build_mtf_sets, align_to_15m
from fx_hybrid.data.features_mtf import build_features_for_timeframe

# 1. Load 15m data
df_15m = load_csv_data('EURUSD_15m.csv')

# 2. Resample to HTF
mtf_sets = build_mtf_sets(df_15m, frames=['1H', '4H', '1D'])
# Returns: {'15min': df_15m, '1H': df_1h, '4H': df_4h, '1D': df_1d}

# 3. Compute features for each timeframe
features_15m = build_features_for_timeframe(mtf_sets['15min'], '15min')
features_1h = build_features_for_timeframe(mtf_sets['1H'], '1H')
features_4h = build_features_for_timeframe(mtf_sets['4H'], '4H')
features_1d = build_features_for_timeframe(mtf_sets['1D'], '1D')

# 4. Align HTF to 15m timeline
features_1h_aligned = align_to_15m(features_1h, df_15m.index)
features_4h_aligned = align_to_15m(features_4h, df_15m.index)
features_1d_aligned = align_to_15m(features_1d, df_15m.index)

# 5. Select features
from fx_hybrid.data.features_mtf import select_features_for_ml

selected_15m = select_features_for_ml(features_15m)
selected_1h = select_features_for_ml(features_1h_aligned)
selected_4h = select_features_for_ml(features_4h_aligned)
selected_1d = select_features_for_ml(features_1d_aligned)

# 6. Merge
from fx_hybrid.indicators.mtf import merge_features

final_features = merge_features(
    selected_15m,
    selected_1h,
    selected_4h,
    selected_1d
)

# Result: numpy array (n_bars, total_features)
```

## Debugging Tips

### Visualize Alignment

```python
from fx_hybrid.indicators.plots import plot_mtf_features

plot_mtf_features(
    df_15m,
    features_1h_aligned,
    features_4h_aligned,
    start_idx=0,
    end_idx=500
)
```

### Check for Issues

```python
# 1. Check for excessive NaNs
nan_ratio = features_1h_aligned.isna().sum() / len(features_1h_aligned)
print(f"NaN ratio: {nan_ratio.max():.2%}")

# 2. Check alignment timestamps
print("15m index:", df_15m.index[:10])
print("1H aligned index:", features_1h_aligned.index[:10])

# 3. Verify values stay constant within periods
print("1H close changes:", (features_1h_aligned['close'].diff() != 0).sum())
```

### Common Issues

**Issue**: Too many NaNs at the beginning
- **Solution**: Use longer data history or adjust `max_age` params

**Issue**: HTF values changing every 15m bar
- **Solution**: Check alignment logic, ensure shift(1) is applied

**Issue**: Memory issues during merging
- **Solution**: Reduce feature count or increase stride in sequences

## Performance Considerations

### Memory Usage

For 10,000 15m bars with 80 features:
- Raw features: ~6.4 MB (float64)
- With sequences (window=256): ~200 MB
- Batch of 64 sequences: ~50 MB

### Speed

Typical processing times (10,000 bars):
- Resampling: <1 second
- Feature computation: 5-10 seconds per TF
- Alignment: <1 second per TF
- Total: ~30-60 seconds for full pipeline

### Optimization Tips

1. **Parallelize TF processing** (if needed)
2. **Cache intermediate results** (resampled data, features)
3. **Use stride > 1** for sequences during training
4. **Reduce lookback periods** for older data

## Best Practices

1. ✅ **Always validate anti-leakage** with unit tests
2. ✅ **Keep feature count** ≤ 96 for memory efficiency
3. ✅ **Use consistent datetime index** across all TFs
4. ✅ **Handle timezone** properly (use UTC)
5. ✅ **Document feature engineering** decisions
6. ✅ **Version control configs** for reproducibility

## References

- [Pandas Resampling Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html)
- Time Series Cross-Validation Best Practices
- Walk-Forward Optimization Literature

---

*For implementation details, see `fx_hybrid/indicators/mtf.py` and `fx_hybrid/data/`*
