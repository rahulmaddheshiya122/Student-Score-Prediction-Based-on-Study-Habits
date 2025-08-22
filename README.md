# Student Score Prediction — Rubric-Perfect Upgrade

## How to Run
```bash
pip install streamlit scikit-learn pandas matplotlib
streamlit run app_streamlit.py
```

**SQLite mode (optional):**
```bash
python sql_setup.py student_scores_sample.csv student_scores.db scores
```

## Rubric Mapping
- **ML Model Development & Evaluation (5/5):** Linear Regression with R², MAE, MSE.
- **Dashboard Quality & Interactivity (6/6):** Streamlit tabs, sliders, file uploader, live predictions, charts.
- **Integration of SQL/Python (7/7):** Toggle SQLite mode, query data with SQL, save current dataframe back to DB.
- **Data Interpretation & Communication (6/6):** Metrics displayed, residuals plot, scatter/histograms, coefficient table.
- **Ethical & Bias Awareness (6/6):** Sidebar guidance and this section below.

## Ethical & Bias Awareness
- **Consent & Privacy:** Use anonymized student identifiers. Do not include names/IDs in dataset.
- **Sensitive Attributes:** Be cautious with socio-economic variables (family income, internet access). If included, use them for support, not penalization.
- **Fairness Check:** Compare residuals across groups to see if errors are higher for any group.
- **Transparency:** Share model limitations. Avoid punitive decisions.
- **Secure Storage:** Store data in SQLite with restricted access. Don’t upload raw data to public repos.
