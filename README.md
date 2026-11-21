# 🏎️ PitWall AI — Formula 1 Race Strategy Simulator

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/) [![FastF1](https://img.shields.io/badge/Telemetry-FastF1-E10600)](https://docs.fastf1.dev/) [![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end F1 race strategy simulator and optimization dashboard modeling fuel-corrected tyre degradation, dynamic weather transitions, Safety Car / VSC pit windows, and 1,000+ Monte Carlo race simulations.

## Features
- **Telemetry Cleaning (`src/data_loader.py`):** FastF1 Grand Prix lap ingestion with pit-lap/outlier filtering and offline synthetic fallback.
- **Tyre Degradation (`src/tyre_model.py`):** Fuel-burn normalized Linear Regression and Gradient Boosting wear models across Soft, Medium, and Hard compounds.
- **Race Simulator (`src/simulator.py`):** Lap-by-lap race engine tracking tyre wear, fuel mass, weather grip penalties (`src/weather.py`), and SC/VSC pit discounts (`src/safety_car.py`).
- **Strategy Optimiser (`src/optimizer.py`):** Evaluates legal 1/2/3-stop strategies under FIA two-compound rules and computes pit-window heatmaps.
- **Monte Carlo Engine (`src/monte_carlo.py`):** Runs 1,000+ paired stochastic races for Win and Podium probabilities.

## Quickstart
```bash
git clone [https://github.com/Kanak0091/pitwall-ai.git](https://github.com/Kanak0091/pitwall-ai.git) && cd pitwall-ai
pip install -r requirements.txt
streamlit run app.py
pytest -v
```

## Author
**Kanak Sachin Chaudhary** — [@Kanak0091](https://github.com/Kanak0091)
