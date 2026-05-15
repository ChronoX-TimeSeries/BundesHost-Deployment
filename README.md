# 🏔️ BundesHost — Tourism Forecasting & Hosting Capacity Analysis

A data-driven project for forecasting tourism demand across German states and exploring how data can support event hosting decisions.

This project is based on official German tourism data from **DESTATIS** and focuses on real-world applicability.

---

## 📌 Project Overview

Tourism demand varies significantly across regions and seasons.  
This project aims to:

- Forecast monthly tourist arrivals for all 16 German states  
- Compare different modeling approaches  
- Explore how forecasting can support **event planning and capacity decisions**

The final outcome is an interactive **Streamlit dashboard** that allows users to simulate event scenarios and evaluate feasibility.

---

## 📊 Data

- Source: DESTATIS (German Federal Statistical Office)  
- Platform: GENESIS-Online (https://www-genesis.destatis.de)  
- Dataset Code: **45412-0025**  

Coverage:
- 16 German federal states  
- Monthly frequency  
- ~30 years of tourism data  

Main variables:
- `date`
- `state`
- `arrivals`
- `overnight`

---
## 📸 Demo

### Dashboard

<p align="center">
  <img src="images/dashboard.png" width="700">
  <br>
  <em>Interactive Streamlit dashboard for exploring forecasts and event scenarios.</em>
</p>

---

### Forecast Example (Berlin)

<p align="center">
  <img src="images/deployment_forecast_berlin.png" width="600">
  <br>
  <em>Example forecast for Berlin, showing model predictions in a deployment setting.</em>
</p>

---

### Best Model by State (All Models)

<p align="center">
  <img src="images/best_model_map_germany.png" width="500">
  <br>
  <em>Best-performing model for each German state based on lowest MAPE across all evaluated approaches.</em>
</p>

---

### Model Performance & Stability

<p align="center">
  <img src="images/model_performance_stability.png" width="600">
  <br>
  <em>Comparison of forecast accuracy (MAPE) across models, highlighting both average performance and variability across German states.</em>
</p>

---

### Best Model per State (SARIMA vs SARIMAX)

<p align="center">
  <img src="images/best_model_sarima_vs_sarimax_map.png" width="500">
  <br>
  <em>Best-performing model selected for each German state between SARIMA and SARIMAX based on lowest MAPE.</em>
</p>

---

## 🔍 Project Workflow

### 1. Data Understanding & Preparation
- Data cleaning and restructuring  
- Time index handling  
- Aggregation per state  

### 2. Exploratory Data Analysis (EDA)
- Trend analysis  
- Seasonality detection  
- State-level comparison  
- Impact of COVID period  

### 3. Time Series Modeling

We explored three families of models:

---

#### 🟩 Statistical Models
- SARIMA  
- SARIMAX  

✔ Capture trend and seasonality explicitly  
✔ Strong and stable performance  

---

#### 🟦 Machine Learning
- XGBoost  

✔ Uses lag-based features  
✔ Captures non-linear relationships  

---

#### 🟪 Deep Learning
- Simple RNN  
- RNN (seasonal features)  
- LSTM  
- LSTM (seasonal features)  

✔ Learn temporal patterns from sequences  
✔ Seasonal variants include cyclical encoding of time  

---

## ⚖️ Model Comparison

- Models were trained and evaluated for each state  
- Performance measured using:
  - MAPE  
  - MAE  

### Key findings:

- **Statistical models (SARIMA/SARIMAX)** provide the most stable performance  
- Deep learning models can perform well in some states, but are less consistent  
- XGBoost is competitive but not consistently superior  

👉 The **best statistical model is selected per state**

---

## 🚀 Deployment (Streamlit App)

An interactive dashboard allows users to:

- Select a German state  
- Choose event size and timing  
- Generate forecasts  
- Evaluate feasibility using a custom metric  

---

### 🧮 Hosting Capacity Score (HCS)

A simple score based on:

- Forecasted tourist demand  
- Event size  

Used to estimate whether a region can handle a given event.

---

## 🎯 Key Features

- 📈 Multi-step time series forecasting  
- 🧠 Model comparison across paradigms  
- 🗺️ State-level analysis  
- ⚡ Interactive Streamlit application  
- 🎯 Decision-support perspective (beyond forecasting)

---

## 🧠 Insights

- Tourism in Germany shows strong yearly seasonality  
- Classical time-series models remain highly effective  
- Model performance varies significantly across states  
- Forecasting can be extended into **decision-support systems**

---

## 🔮 Future Work

### Modeling Improvements
- Include external features (weather, holidays, events)  
- Explore ensemble / hybrid models  

### Product Extension
- Extend to a **decision-support system**  
- Incorporate:
  - accommodation capacity  
  - infrastructure  
  - transportation  

### Engineering
- Real-time or higher-frequency data  
- Scalable deployment  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Statsmodels  
- Scikit-learn  
- XGBoost  
- TensorFlow / Keras  
- Plotly  
- Streamlit  

---

## ⚙️ Setup & Requirements

### Python Environment
- Python 3.11.3 (recommended via `pyenv`)

---

### 🔧 Installation

```bash
# 1. Set Python version (recommended via pyenv)
pyenv local 3.11.3

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### ⚙️ Regenerate Models (optional)

Trained models and the best-model selection are committed to the repo,
so the Streamlit app works immediately after cloning.

If you want to retrain (e.g. after updating the data), run:

```bash
# 1. Evaluate both SARIMA and SARIMAX on the test set, pick the best per state
python -m modeling.evaluate

# 2. Retrain the best model on the full dataset and save it
python -m modeling.train
```
---

### ▶️ Run the App

```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

The project is organized in a modular way, separating data processing, modeling, and deployment:

```
BundesHost-Deployment/
│
├── app/                       # Streamlit dashboard
│   └── streamlit_app.py
│
├── data/
│   ├── raw/                   # Original Destatis CSV
│   ├── processed/             # Cleaned dataset (tourism_long.csv)
│   └── 2_hoch.geo.json        # GeoJSON for Germany map
│
├── modeling/                  # Core modeling package
│   ├── config.py              # Paths, states, COVID period, model orders
│   ├── data_pipeline.py       # Load + reshape raw Destatis CSV
│   ├── feature_engineering.py # State time series + COVID dummy
│   ├── train.py               # Train SARIMA + SARIMAX per state
│   ├── evaluate.py            # Evaluate + select best model → JSON
│   └── predict.py             # Load best model + forecast
│
├── models/                    # Saved models + best_models.json (gitignored)
│
├── notebooks/                 # EDA & experimentation
│   ├── 00_data_understanding.ipynb
│   ├── 01_eda_tourism_germany.ipynb
│   └── 02_time_series_modeling.ipynb
│
├── images/                    # Figures for README
│
├── Makefile                   # Setup shortcuts
├── requirements.txt           # Project dependencies
├── runtime.txt                # Deployment environment (e.g. Streamlit Cloud)
│
└── README.md
```

---

## 👥 Team

**ChronoX — Time Series Capstone Project**  
SPICED Academy · 2026  

- **Nazila Fazeli** — Data Science & Machine Learning  
  📍 Hamburg  
  🔗 https://www.linkedin.com/in/nazila-fazeli/ 

- **Luka Androcec** — UX/UI & Customer Experience  
  📍 Berlin  
  🔗 https://www.linkedin.com/in/lukaandrocec/ 

- **Emna Zayani** — Data & Business Intelligence  
  📍 Berlin  
  🔗 https://www.linkedin.com/in/emna-zayani/  
---

## 📎 Links

- GitHub Repository: *(https://github.com/ChronoX-TimeSeries/BundesHost-Deployment)*  
- Streamlit App: *(https://chronox-timeseries.streamlit.app)*  

---

## 💬 Final Note

This project demonstrates how time series forecasting can move beyond prediction and support real-world decision making.