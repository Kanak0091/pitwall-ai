<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B0E14,50:990000,100:E10600&height=200§ion=header&text=PITWALL%20AI&fontSize=54&fontColor=F8FAFC&fontAlignY=38&desc=Formula%201%20Race%20Strategy%20Simulator&descAlignY=62&descSize=18" width="100%" alt="PitWall AI Header"/>

<a href="https://github.com/Kanak0091/pitwall-ai"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=16&duration=2800&pause=800&color=E10600¢er=true&vCenter=true&width=850&lines=BOX+BOX+%E2%80%A2+Fuel-Corrected+Tyre+Degradation+ML;Dynamic+Weather+%26+Safety+Car+Pit-Window+Optimizer;1%2C000%2B+Paired+Monte+Carlo+Race+Simulations" alt="PitWall Telemetry Ticker" /></a>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-Telemetry%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-Degradation%20ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/) [![FastF1](https://img.shields.io/badge/Telemetry-FastF1-E10600?style=for-the-badge)](https://docs.fastf1.dev/) [![Tests](https://img.shields.io/badge/Tests-9%20Passing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)

</div>

---

## 🏁 Overview

**PitWall AI** is a full-stack Formula 1 race strategy simulation, optimization, and decision-support engine built in Python. Inspired by real-world Grand Prix pit-wall telemetry tools, it ingests historical F1 lap data via **FastF1**, isolates pure tyre degradation from fuel-burn pace gains, simulates full race distances lap-by-lap under dynamic weather and Safety Car neutralizations, and runs **1,000+ paired Monte Carlo simulations** to quantify strategy win probabilities.

---

## ✨ Architecture & Core Modules

| Module | File | Responsibility |
| :--- | :--- | :--- |
| **Telemetry Data Loader** | `src/data_loader.py` | Fetches real Grand Prix race sessions via **FastF1**, filters pit in/out laps, Safety Car laps, and >107% outliers, and provides an offline synthetic generator for instant execution. |
| **Tyre Degradation Engine** | `src/tyre_model.py` | Applies a **~0.032s/lap fuel-burn normalization** before training **Linear Regression** or **Gradient Boosting** models for `SOFT`, `MEDIUM`, and `HARD` compounds. |
| **Weather Transition Model** | `src/weather.py` | Simulates `dry`, `damp`, and `wet` track states, computes compound-to-surface grip penalties, and triggers automatic pit switches to `INTERMEDIATE` or `WET` tyres. |
| **Safety Car & VSC Engine** | `src/safety_car.py` | Models randomized or user-triggered `SC` and `VSC` periods that cut pit-lane time loss by 35–50% and generates real-time **PIT NOW vs STAY OUT** pit-wall advice. |
| **Lap-by-Lap Race Simulator** | `src/simulator.py` | Executes full race distances lap-by-lap, tracking stint age, compound pace offsets, fuel weight, pit-lane loss, and neutralization multipliers. |
| **Multi-Stop Optimizer** | `src/optimizer.py` | Evaluates valid 1-stop, 2-stop, and 3-stop strategies (enforcing FIA two-compound dry rules), ranks candidates, and computes optimal pit-window heatmaps. |
| **Monte Carlo Simulator** | `src/monte_carlo.py` | Runs **1,000+ paired stochastic trials** (Common Random Numbers) across randomized Safety Cars, rain showers, and lap-time noise to calculate Win (%) and Podium (%) odds. |
| **Interactive Telemetry UI** | `app.py` | Dark F1 telemetry dashboard built with **Streamlit** and **Plotly**, featuring Pirelli compound badges, degradation curves, lap-by-lap traces, and pit-window heatmaps. |

---

## 🧠 Mathematical & Domain Modeling

### 1. Separating Fuel Burn from True Tyre Wear
A Formula 1 car starts a Grand Prix with ~100–110 kg of fuel and burns ~1.7 kg per lap, making the car naturally **~0.032 seconds faster each lap** as mass drops. PitWall AI normalizes every lap to a **zero-fuel baseline** first:

$$\\text{ZeroFuelLapTime}_i = \\text{RawLapTime}_i - 0.032 \\times (\\text{TotalLaps} - \\text{Lap}_i)$$

### 2. Lap-by-Lap Pace Equation
On every simulated lap $t \\in [1, T]$, predicted green-flag lap time is evaluated as:

$$\\text{LapTime}(t) = \\text{BasePace}_{c} + \\text{Degradation}_{c}(\\text{TyreAge}_t) + 0.032(T - t) + \\text{WeatherPenalty}(c, w_t) + \\epsilon_t$$

- Under **VSC**, lap times are scaled by `1.18x` and net pit-stop loss drops by **35%**.
- Under a full **Safety Car (SC)**, lap times are scaled by `1.38x` and net pit-stop loss drops by **50%**, creating opportunistic pit windows.

### 3. Paired Monte Carlo Sampling
Trial $k \\in [1, N]$ generates **one shared weather timeline** and **one shared Safety Car schedule**, evaluating every candidate strategy against identical race chaos before computing Win (%) and Podium (%) probabilities.

---

## 📂 Project Structure

```text
pitwall-ai/
├── assets/
│   ├── header_banner.svg   # Animated F1 telemetry header banner
│   └── footer_banner.svg   # Chequered-flag telemetry footer banner
├── src/
│   ├── data_loader.py      # FastF1 session loader, cleaner & synthetic fallback
│   ├── tyre_model.py       # Fuel-burn correction & Linear/GBR degradation models
│   ├── weather.py          # Dry/damp/wet grip penalties & wet-tyre triggers
│   ├── safety_car.py       # SC/VSC pit-loss discounts & pit-now recommendations
│   ├── simulator.py        # Core lap-by-lap race execution engine
│   ├── optimizer.py        # 1/2/3-stop strategy search & pit-window heatmaps
│   └── monte_carlo.py      # 1,000+ stochastic race simulation engine
├── tests/
│   └── test_pitwall.py     # Pytest unit test suite covering all modules
├── .streamlit/
│   └── config.toml         # Custom F1 dark telemetry theme configuration
├── app.py                  # Interactive Streamlit + Plotly dashboard
├── requirements.txt
├── pytest.ini
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Installation & Quickstart

```bash
git clone https://github.com/Kanak0091/pitwall-ai.git
cd pitwall-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
pytest -v
```

---

## 📊 Sample Optimization Output

```text
Rank  Strategy                        Stops  Compounds               TotalTime_s  DeltaToBest_s
   1  M-H (1-Stop | Lap 24)               1  MEDIUM -> HARD             5368.420          0.000
   2  S-H-M (2-Stop | Laps 15, 37)        2  SOFT -> HARD -> MEDIUM     5370.185          1.765
   3  M-H-S (2-Stop | Laps 21, 43)        2  MEDIUM -> HARD -> SOFT     5371.040          2.620
```

---

## ⚠️ Model Limitations & Roadmap

- **Single-Car Pace Approximation:** Models clean-air race time without 20-car traffic gaps, DRS trains, or dirty-air aero loss when rejoining behind slower cars.
- **Smooth Wear Curves:** Uses linear and shallow gradient-boosted degradation curves rather than thermal tyre-surface modeling (graining or sudden blistering cliffs).
- **Planned Enhancements:** Multi-car field simulation for undercut/overcut track-position battles, track rubber evolution over race distance, and circuit-specific pit-loss presets.

---

## 👤 Author

**Kanak Sachin Chaudhary**  
BCA (AI & Data Science) — K.R. Mangalam University  
GitHub: [@Kanak0091](https://github.com/Kanak0091)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:E10600,50:990000,100:0B0E14&height=120§ion=footer&text=BOX%20BOX%20%E2%80%A2%20Kanak%20Sachin%20Chaudhary&fontSize=20&fontColor=F8FAFC&fontAlignY=68" width="100%" alt="PitWall AI Footer"/>

</div>
