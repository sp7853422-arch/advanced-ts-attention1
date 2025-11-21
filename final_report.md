#  Final Report: Advanced Time Series Forecasting

## 1. Introduction
We forecast electricity load using the UCI dataset (2011–2014).  
Goal: compare classical baselines with deep learning models (LSTM+Attention, Transformer).

## 2. Dataset & Preprocessing
- Resampled to hourly frequency.
- Selected 10 clients.
- Chronological split: 70% train, 15% validation, 15% test.
- Normalized with MinMaxScaler.

## 3. Methodology
- **Baselines:** Naive persistence, SARIMA, Prophet.
- **Deep Learning:** LSTM+Attention, Transformer.
- **Hyperparameter Tuning:** KerasTuner Bayesian Optimization.

## 4. Results
| Model              | MAE (Test) | RMSE (Test) |
|--------------------|------------|-------------|
| Naive Baseline     | …          | …           |
| SARIMA / Prophet   | …          | …           |
| LSTM + Attention   | …          | …           |
| Transformer        | …          | …           |
| Tuned LSTM+Attn    | …          | …           |

## 5. Interpretability
- Attention weights highlight daily/weekly cycles.
- Transformer captures long-term dependencies.

## 6. Conclusion
- Attention-based LSTM improves interpretability and accuracy.
- Transformer achieves competitive performance.
- Hyperparameter tuning further boosts results.
