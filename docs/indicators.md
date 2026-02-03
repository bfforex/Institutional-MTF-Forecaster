# Indicators Specification

## Fair Value Gaps (FVG)

### Overview

Fair Value Gaps (FVGs) are areas where price moved so quickly that it left an "inefficiency" or gap in the market. These gaps often act as magnets for price, as the market seeks to fill these imbalances.

### Detection Logic

#### Bullish FVG

A bullish FVG occurs when:
- `low[i+1] > high[i-1]`
- Candle `i` is a displacement bar (high momentum)
- Gap size ≥ minimum pip threshold

```
Price
  ↑
  |     ┌─┐  ← Candle i+1 (low > prev high)
  |     └─┘
  |  GAP  ← Fair Value Gap
  |     ┌─┐  ← Candle i (displacement)
  |     └─┘
  |     ┌─┐  ← Candle i-1
  |     └─┘
  └──────────→ Time
```

#### Bearish FVG

A bearish FVG occurs when:
- `high[i+1] < low[i-1]`
- Candle `i` is a displacement bar
- Gap size ≥ minimum pip threshold

### Displacement Filter

To ensure quality FVGs, we filter for displacement candles:

**ATR Method** (default):
```
candle_range[i] >= k × ATR[i]
```
where k = 1.5 to 2.5 (configurable)

**Z-Score Method**:
```
zscore = (range[i] - mean(range)) / std(range)
zscore >= k
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 2.0 | Displacement multiplier |
| `atr_len` | 14 | ATR calculation period |
| `min_gap_pips` | 2.0 | Minimum gap size in pips |
| `max_age` | 200 | Maximum age before expiry |
| `pip_size` | 0.0001 | Pip size for instrument |

### FVG Features

The detector outputs the following features per bar:

1. **Detection Flags**
   - `fvg_has_bull`: Boolean, true if bullish FVG formed
   - `fvg_has_bear`: Boolean, true if bearish FVG formed

2. **Gap Metrics**
   - `fvg_gap_pips`: Gap size in pips
   - `fvg_gap_zscore`: Normalized gap size
   
3. **Age & Status**
   - `fvg_age_bars`: Bars since FVG formation
   - `fvg_partial_fill_ratio`: Fraction of gap filled (0-1)
   - `fvg_context`: "fresh", "partially_filled", or "mitigated"

4. **Distance Features**
   - `dist_to_nearest_bull_fvg`: Distance in pips to nearest bullish FVG
   - `dist_to_nearest_bear_fvg`: Distance in pips to nearest bearish FVG

### Usage Example

```python
from fx_hybrid.indicators import detect_fvg

# Detect FVGs
df_with_fvg = detect_fvg(
    df,
    k=2.0,
    atr_len=14,
    min_gap_pips=2.0,
    max_age=200,
    displacement='atr'
)

# Check for recent bullish FVG
recent_bull_fvg = df_with_fvg['fvg_has_bull'].iloc[-10:].any()
```

---

## Smart Money Concepts (SMC)

### Overview

Smart Money Concepts (SMC) analyze institutional order flow and market structure. These indicators help identify where "smart money" (banks, institutions) are likely positioned.

### Components

#### 1. Swing Structure

**Swing Highs**:
- `high[i]` greater than `w` neighbors on both sides
- Minimum excursion ≥ `min_exc_atr × ATR`

**Swing Lows**:
- `low[i]` lower than `w` neighbors on both sides
- Minimum excursion ≥ `min_exc_atr × ATR`

```
        H
       / \      ← Swing High
      /   \
     /     \
    L       L   ← Not swing lows (too close)
```

#### 2. Break of Structure (BOS)

A BOS occurs when price breaks a recent swing high/low in the direction of the trend:

**Bullish BOS**: Close breaks above last swing high (continuation of uptrend)
**Bearish BOS**: Close breaks below last swing low (continuation of downtrend)

#### 3. Change of Character (CHOCH)

A CHOCH signals a potential trend reversal:

**Bullish CHOCH**: Close breaks above swing high after bearish trend
**Bearish CHOCH**: Close breaks below swing low after bullish trend

#### 4. Order Blocks (OB)

Order Blocks are the last opposite-colored candle before a strong displacement move (BOS):

**Bullish OB**: Last bearish candle before bullish BOS
**Bearish OB**: Last bullish candle before bearish BOS

These zones represent where institutions placed large orders and may defend these levels.

#### 5. Liquidity Sweeps

Occur when price briefly penetrates equal highs/lows (liquidity zones) but closes back inside:

```
Price
  ↑
  |    ╱╲  ← Wick sweeps equal highs
  |   ╱  ╲
  |══════════ ← Equal highs (liquidity)
  |  █  █
  └──────────→ Time
```

#### 6. Premium/Discount Zones

Based on the current swing range:

- **Premium**: Price > 61.8% of range (expensive)
- **Equilibrium**: 38.2% - 61.8% (fair value)
- **Discount**: Price < 38.2% of range (cheap)

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swing_w` | 3 | Swing window (bars on each side) |
| `min_exc_atr` | 0.5 | Minimum swing excursion in ATR |
| `equal_band_pips` | 2.0 | Band for equal highs/lows |
| `wick_ratio` | 0.6 | Minimum wick-to-body ratio for sweeps |
| `disp_k` | 2.0 | Displacement multiplier for OBs |
| `max_ob_boxes` | 3 | Maximum number of active OBs |

### SMC Features

Output features:

1. **Trend & Structure**
   - `trend_bias`: +1 (bullish), 0 (neutral), -1 (bearish)
   - `bos_flag`: Boolean, BOS occurred
   - `choch_flag`: Boolean, CHOCH occurred
   - `bos_age`: Bars since last BOS
   - `choch_age`: Bars since last CHOCH

2. **Order Blocks**
   - `ob_up_center`: Center price of bullish OB
   - `ob_up_width`: Width of bullish OB
   - `ob_dn_center`: Center price of bearish OB
   - `ob_dn_width`: Width of bearish OB
   - `dist_to_ob_up`: Distance to bullish OB
   - `dist_to_ob_dn`: Distance to bearish OB

3. **Liquidity**
   - `liq_sweep_high`: High sweep occurred
   - `liq_sweep_low`: Low sweep occurred
   - `sweep_age`: Bars since last sweep

4. **Premium/Discount**
   - `pd_zone`: "premium", "equilibrium", or "discount"
   - `pd_level`: Percentage within range (0-100)
   - `dist_to_eq_50`: Distance to 50% equilibrium
   - `dist_to_swing`: Distance to nearest swing boundary

### Usage Example

```python
from fx_hybrid.indicators import compute_smc_features

# Compute all SMC features
df_with_smc = compute_smc_features(
    df,
    swing_w=3,
    min_exc_atr=0.5,
    equal_band_pips=2.0,
    wick_ratio=0.6,
    disp_k=2.0,
    max_ob_boxes=3
)

# Check if in discount zone with bullish bias
is_discount = df_with_smc['pd_zone'].iloc[-1] == 'discount'
is_bullish = df_with_smc['trend_bias'].iloc[-1] > 0
```

---

## Market Regime Detection

### Overview

Classifies market conditions for adaptive strategy selection.

### Regimes

1. **Trending Up**: ADX > threshold, EMA20 > EMA50
2. **Trending Down**: ADX > threshold, EMA20 < EMA50
3. **Ranging**: ADX < threshold
4. **High Volatility**: ATR > 75th percentile
5. **Low Volatility**: ATR < 75th percentile

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 100 | Lookback period for regime |
| `adx_threshold` | 25 | ADX threshold for trending |

### Features

- `regime`: Categorical regime label
- `regime_numeric`: Numeric encoding (-2 to +2)
- `volatility_regime`: "high_vol" or "low_vol"
- `adx`: ADX value
- `atr_percentile`: ATR percentile ranking

### Usage Example

```python
from fx_hybrid.indicators import detect_regime

df_with_regime = detect_regime(
    df,
    lookback=100,
    adx_threshold=25
)

# Check current regime
current_regime = df_with_regime['regime'].iloc[-1]
```

---

## Visualization

Use the plotting utilities to visualize indicators:

```python
from fx_hybrid.indicators.plots import plot_fvg, plot_smc, plot_regime

# Plot FVG
plot_fvg(df_with_fvg, start_idx=0, end_idx=500)

# Plot SMC
plot_smc(df_with_smc, start_idx=0, end_idx=500)

# Plot regime
plot_regime(df_with_regime, start_idx=0, end_idx=500)
```

---

## Tuning Guidelines

### For Different Currency Pairs

**Major Pairs (EURUSD, GBPUSD)**:
- `min_gap_pips`: 2.0 - 3.0
- `k`: 2.0 - 2.5
- `equal_band_pips`: 2.0

**Volatile Pairs (GBPJPY, AUDJPY)**:
- `min_gap_pips`: 5.0 - 8.0
- `k`: 2.5 - 3.0
- `equal_band_pips`: 5.0

**Exotic Pairs**:
- Increase all pip-based thresholds by 2-3x
- Use higher ATR multipliers

### For Different Timeframes

**Higher Timeframes (4H, 1D)**:
- Increase `max_age` (500-1000 bars)
- Increase `min_exc_atr` (0.8-1.2)

**Lower Timeframes (5m, 1m)**:
- Decrease `max_age` (50-100 bars)
- Decrease `min_exc_atr` (0.3-0.5)
- Stricter displacement filters

---

## Best Practices

1. **Always validate indicators visually** before trading
2. **Backtest with transaction costs** included
3. **Use confluence of multiple indicators** for higher probability setups
4. **Adjust parameters based on market volatility**
5. **Monitor indicator performance** and recalibrate periodically

---

*For implementation details, see the source code in `fx_hybrid/indicators/`*
