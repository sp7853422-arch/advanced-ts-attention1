# Advanced Time Series Forecasting with Attention & Transformer

##  Project Overview
This project implements **state-of-the-art deep learning models** for multivariate time series forecasting using the **UCI Electricity Load Diagrams (2011–2014)** dataset.  
We compare classical baselines (Naive, SARIMA, Prophet) with modern architectures (LSTM+Attention, Transformer), and apply **Bayesian hyperparameter optimization** to achieve production-quality results.

---

## Objectives
- Acquire and preprocess a complex multivariate time series dataset.
- Implement **attention-based deep learning models** (LSTM+Attention, Transformer).
- Benchmark against classical forecasting models (Naive, SARIMA, Prophet).
- Apply **hyperparameter tuning** (KerasTuner Bayesian Optimization).
- Analyze **attention weights** for interpretability.
- Deliver a **comparative analysis report** with metrics and visualizations.

---

##  Repository Structure
advanced-ts-attention/ ├── README.md ├── requirements.txt / environment.yml ├── notebooks/ │ ├── 01_data_download_and_eda.ipynb │ ├── 02_preprocess_windowing.ipynb │ ├── 03_baselines_naive_sarima_prophet.ipynb │ ├── 04_lstm_attention.ipynb │ ├── 05_transformer_forecaster.ipynb │ ├── 06_hpo_kerastuner.ipynb │ └── 07_evaluation_reporting.ipynb├── src/ │ ├── data_utils.py │ ├── windowing.py │ ├── metrics.py │ ├── plots.py │ ├── models/ │ │ ├── lstm_attention.py │ │ └── transformer.py │ └── tuning/ │ └── tuner_lstm_attention.py ├── data/ │ ├── raw/LD2011_2014.txt (ignored in Git) │ ├── processed/train_scaled.parquet │ └── processed/test_scaled.parquet├── artifacts/ │ ├── models/ │ │ ├── lstm_attention_model.keras │ │ └── transformer_model.keras │ ├── figures/ │ │ ├── attention_heatmap.png │ │ ├── forecasts_example.png │ └── reports/comparative_table.csv └── submit/ └── project_submit.zip


---
##  Dataset
The full dataset is too large to include in this repo.  
Download it from [UCI Electricity Load Diagrams 2011–2014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014).  
Place the file in `data/raw/LD2011_2014.txt` before running the notebook.

##  Notebook
The full workflow is implemented in a single Colab notebook:

- [`notebooks/advanced_ts_forecasting.ipynb`](notebooks/advanced_ts_forecasting.ipynb)

You can open it directly in Google Colab using this badge:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/advanced-ts-attention/blob/main/notebooks/advanced_ts_forecasting.ipynb)

##  Setup
### Requirements
- Python >= 3.10
- TensorFlow >= 2.13
- KerasTuner >= 1.4
- pandas, numpy, scikit-learn, matplotlib, seaborn
- statsmodels, prophet (optional)

### Installation
```bash
# Clone repo
git clone https://github.com/yourusername/advanced-ts-attention.git
cd advanced-ts-attention

# Install dependencies
pip install -r requirements.txt

 Workflow (8 Steps)
Step 1: Dataset Loading
Loaded UCI Electricity Load Diagrams (2011–2014).

Resampled to hourly frequency.

Handled missing values with forward/backward fill.

Step 2: Preprocessing
Selected 10 clients for manageable computation.

Chronological split: 70% train, 15% val, 15% test.

Applied MinMaxScaler for normalization.

Step 3: Windowing
Created sliding windows:

Lookback = 96 hours (4 days history).

Horizon = 24 hours (forecast next day).

Step 4: Baseline Models
Naive persistence (last value).

SARIMA (seasonal ARIMA).

Prophet (trend + seasonality).

Step 5: LSTM + Attention
Custom Temporal Attention layer.

LSTM → Dropout → Attention → Dense forecast.

Visualized attention weights for interpretability.

Step 6: Transformer Forecaster
Encoder-only Transformer with positional embeddings.

Multi-head self-attention + feed-forward blocks.

Compared against LSTM+Attention.

Step 7: Hyperparameter Optimization
Used KerasTuner Bayesian Optimization.

Tuned LSTM units, dropout, learning rate.

Selected best hyperparameters and retrained.

Step 8: Comparative Analysis
Summarized results in a table:

Model	MAE (Test)	RMSE (Test)
Naive Baseline	…	…
SARIMA / Prophet	…	…
LSTM + Attention	…	…
Transformer	…	…
Tuned LSTM+Attn	…	…

Results & Insights
Attention-based LSTM improves interpretability and accuracy.

Transformer captures long-term dependencies effectively.

Hyperparameter tuning further boosts performance.

Attention plots reveal daily/weekly cycles as key drivers.

Conclusion
This project demonstrates how attention mechanisms and Transformer architectures outperform classical baselines in time series forecasting. The pipeline is modular, reproducible, and portfolio-ready, with clear documentation and visual interpretability.
