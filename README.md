# Virtual Climate Modeling

An interactive climate analytics project exploring historical global temperature anomalies, comparing simple baselines with machine-learning models, inspecting model disagreement, and explaining fitted model behavior with SHAP.

> **Live demo:** [Open the Streamlit app](https://virtual-climate-modeling-vmgvxdngsygb2njabqytlg.streamlit.app/)  
> **Source code:** [GitHub repository](https://github.com/Nijaswini/virtual-climate-modeling)

## Project overview

The project uses NASA GISTEMP global temperature anomaly data to build a Python analysis pipeline and a public Streamlit dashboard. It covers data preparation, time-aware model evaluation, explainable AI, model disagreement, and an illustrative linear extrapolation.

The focus is on **transparent evaluation**, not claiming that a more complex model is automatically better.

## Dashboard features

- **Historical Temperature:** Explore the annual global temperature anomaly series.
- **Model Comparison:** Compare a previous-year baseline, Linear Regression, and Random Forest.
- **Time-Series Validation:** Evaluate models using chronological splits rather than random shuffling.
- **Uncertainty Analysis:** Inspect disagreement between model predictions.
- **Explainable AI (SHAP):** Explore which features influenced the fitted model.
- **Future Projection:** View a simple linear extrapolation with clear scientific caveats.
- **Project Files:** Review selected outputs and project artifacts.

## Results at a glance

### Chronological held-out test set (1997–2025)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Previous-year baseline | 0.0990 | 0.1163 | 0.7418 |
| Linear Regression | 0.1644 | 0.1913 | 0.3015 |
| Random Forest | 0.4366 | 0.4933 | -3.6417 |

Lower MAE and RMSE indicate smaller prediction errors. In this experiment, the simple previous-year baseline outperformed both tested machine-learning models on the held-out period.

### Average 5-fold time-series validation

| Model | Mean MAE | Mean RMSE |
|---|---:|---:|
| Previous-year baseline | 0.0945 | 0.1137 |
| Linear Regression | 0.1158 | 0.1402 |
| Random Forest | 0.2032 | 0.2357 |
| Equal-weight ensemble | 0.1549 | 0.1842 |

These are results from this project's current experimental setup, not a claim that the same ranking will hold for every dataset or evaluation period. Errors are in the dataset's temperature-anomaly units (°C).

## Visual outputs

Generated charts include:

- `outputs/global_temperature_trend.png`
- `outputs/model_predictions.png`
- `outputs/shap_feature_importance.png`
- `outputs/future_projection.png`

Open the corresponding files in the `outputs/` directory to view the charts.

## Technology stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine learning:** Scikit-learn
- **Explainability:** SHAP
- **Interactive dashboard:** Streamlit
- **Deployment:** Streamlit Community Cloud

## Project structure

```text
virtual-climate-modeling/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── notebooks/
├── outputs/
└── src/
    ├── download_data.py
    ├── explore_data.py
    ├── train_model.py
    ├── validate_models.py
    ├── ensemble_model.py
    ├── uncertainty_analysis.py
    ├── explainability.py
    └── future_projection.py
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Nijaswini/virtual-climate-modeling.git
cd virtual-climate-modeling
```

### 2. Create and activate a virtual environment (recommended)

Windows:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal, typically `http://localhost:8501`.

## Data source

NASA GISTEMP global surface temperature analysis:

- [NASA GISS Surface Temperature Analysis](https://data.giss.nasa.gov/gistemp/)
- [Global temperature anomaly table](https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv)

The analysis uses complete annual observations through 2025; partial 2026 data should not be treated as a complete year.

## Scientific limitations

- The previous-year baseline performed better than the tested machine-learning models in the reported evaluation. This is an important finding, not a result to hide.
- The tested feature set is limited and does not represent the full climate system.
- The disagreement between Linear Regression and Random Forest is a **model-disagreement indicator**, not a calibrated prediction interval or a complete measure of climate uncertainty.
- SHAP explains the behavior of the fitted model; it does not establish causal relationships.
- The future projection is a **simple linear extrapolation**, not a validated climate forecast. It does not model emissions scenarios, ocean dynamics, or other physical climate processes.
- Temperature anomalies are relative to the dataset's reference baseline; they are not absolute global temperature readings.

## Possible next improvements

- Test additional time-series baselines and stronger diagnostic checks.
- Investigate model performance across multiple evaluation windows.
- Add formal prediction intervals and calibration evaluation.
- Compare with scenario-based or physically informed climate projections.
- Improve automated testing, reproducibility, and data-version tracking.

## Author

Developed as a climate data analysis and machine-learning portfolio project.

- **Live app:** https://virtual-climate-modeling-vmgvxdngsygb2njabqytlg.streamlit.app/
- **GitHub:** https://github.com/Nijaswini/virtual-climate-modeling
