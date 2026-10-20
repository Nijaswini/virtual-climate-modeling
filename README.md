
# Virtual Climate Modeling

An interactive climate analytics dashboard exploring historical global temperature anomalies, machine-learning model performance, uncertainty, explainable AI, and illustrative future trends.

## Features

- Historical global temperature anomaly analysis using NASA GISTEMP data
- Linear Regression and Random Forest model comparison
- Previous-year baseline for evaluating model performance
- Time-series model evaluation and prediction visualizations
- Model disagreement analysis
- SHAP-based feature importance
- Illustrative linear-trend projections
- Interactive Streamlit dashboard

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- SHAP
- Streamlit

## Run Locally

Clone the repository and enter the project folder.

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

## Data Source

NASA GISTEMP global temperature analysis:
https://data.giss.nasa.gov/gistemp/

## Scientific Limitations

The tested machine-learning models do not necessarily outperform the previous-year baseline. SHAP values explain the fitted model's behavior but do not establish causation. Future projections are simple linear extrapolations, not validated climate forecasts.

## Author

Project developed as a climate data analysis and machine-learning project.
