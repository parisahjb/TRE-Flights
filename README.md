# ✈️ Flight Disruption Regime Dashboard

Interactive Streamlit dashboard accompanying the paper:

> Hajibabaee, P. et al. *Predicting the Unpredictable: Machine Learning Model Failure During Operational Regime Shifts and Early Warning System Development — Evidence from 103.7 Million Flights Across COVID-19 and Financial Crisis.* Transportation Research Part E, 2026.

## 🚀 Live App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📋 Dashboard Tabs

### 📡 Early Warning System
- Weekly ROC-AUC monitoring (855 weeks, 2009–2025)
- CUSUM control chart with ±2σ limits
- Bollinger Band anomaly detection
- Four-tier alert timeline (Green / Yellow / Orange / Red)
- Detected COVID-19 regime shift 9 weeks before WHO declaration

### 💰 Economic Decision Calculator
- Interactive cost matrix (C_d, C_i, π sliders)
- Computes optimal intervention threshold per regime
- Sensitivity analysis across prevention rates
- Confusion matrix breakdown for all four regimes

### 🔬 Regime Mechanism Explorer
- Feature group importance trajectories across four economic periods
- Percentage change from Normal Operations baseline
- DiD causal evidence: Hub vs. Point-to-Point airlines
- ROC-AUC performance table

## 🗂️ Repository Structure
```
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── data/
│   ├── cell12_early_warning_monitoring.csv   # EWS weekly data (855 rows)
│   ├── table2_economic_thresholds.csv        # Regime economic results
│   ├── table3_sensitivity_analysis.csv       # Sensitivity analysis
│   ├── cell9_group_summary.csv               # Feature group importances
│   ├── results_by_period.csv                 # Model performance by regime
│   └── cell13_did_estimates.csv              # DiD causal estimates
└── README.md
```

## ⚙️ Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/flight-disruption-dashboard
cd flight-disruption-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Data

All data files are derived from the BTS On-Time Performance database
(103.7 million U.S. domestic flights, 2009–2025). No raw flight data
is included in this repository — only aggregated model outputs.

## 📄 Citation
```bibtex
@article{hajibabaee2026flight,
  title   = {Predicting the Unpredictable: Machine Learning Model Failure 
             During Operational Regime Shifts},
  author  = {Hajibabaee, Parisa and others},
  journal = {Transportation Research Part E},
  year    = {2026}
}
```

## 🔗 Related

- [Paper (TRE)](https://doi.org/XXXX)
- [Florida Polytechnic University — Data Science](https://floridapoly.edu)
```

---

## File 4: `.gitignore`
```
__pycache__/
*.pyc
*.pyo
.env
.venv
venv/
*.egg-info/
dist/
build/
.DS_Store
*.pkl
*.ipynb_checkpoints
checkpoint_*.csv
```

---

## Step-by-Step: Create the Repository

**1. Create folders**
```
flight-disruption-dashboard/
    app.py
    requirements.txt
    README.md
    .gitignore
    data/
        cell12_early_warning_monitoring.csv
        table2_economic_thresholds.csv
        table3_sensitivity_analysis.csv
        cell9_group_summary.csv
        results_by_period.csv
        cell13_did_estimates.csv
