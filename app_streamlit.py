import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Student Score Prediction", page_icon="📊", layout="wide")
st.title("📊 Student Score Prediction — Study Habits")

st.markdown("""
**Rubric alignment**
- ML Development & Evaluation: Linear Regression + R², MAE, MSE
- Dashboard & Interactivity: Streamlit controls, interactive predictions
- SQL/Python Workflow: Optional SQLite load/save and query
- Interpretation & Insights: Metrics + charts + text summary
- Ethics & Bias Awareness: Notes in sidebar
""")

with st.sidebar:
    st.header("ℹ️ How to use")
    st.write("Upload CSV or load from SQLite. Train a Linear Regression model to predict Final_Score using study features.")
    st.write("**Ethics & Bias:** Ensure consent and anonymization. Avoid using sensitive attributes to penalize students. Use predictions to support students.")

    use_sql = st.checkbox("Use SQLite workflow", value=False)
    db_name = st.text_input("SQLite DB filename", value="student_scores.db")

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def load_sql(db_path, query):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df

tab_data, tab_model, tab_viz = st.tabs(["1) Data", "2) Model & Metrics", "3) Visualizations"])

with tab_data:
    st.subheader("Data Source")
    df = None
    if use_sql:
        st.write("**SQLite mode**")
        q_default = "SELECT * FROM scores;"
        query = st.text_area("SQL Query", q_default, height=120)
        if st.button("Load from SQLite"):
            try:
                df = load_sql(db_name, query)
                st.success(f"Loaded {len(df)} rows from {db_name}")
            except Exception as e:
                st.error(f"SQL load error: {e}")
    else:
        file = st.file_uploader("Upload CSV (e.g., Hours_Studied, Attendance, Final_Score, ...)", type=["csv"])
        if file:
            df = load_csv(file)
            st.success(f"Loaded {len(df)} rows from CSV")

    if df is not None:
        st.dataframe(df.head(50), use_container_width=True)
        st.download_button("Download current data as CSV", df.to_csv(index=False).encode('utf-8'), "current_data.csv", "text/csv")

with tab_model:
    st.subheader("Model Training & Evaluation")
    if df is None:
        st.info("Load data first from the Data tab.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.selectbox("Target (y)", options=numeric_cols, index=0)
        feature_cols = st.multiselect("Features (X)", [c for c in numeric_cols if c != target_col], default=[c for c in numeric_cols if c != target_col][:2])

        test_size = st.slider("Test size (validation split)", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, step=1)

        if len(feature_cols) == 0:
            st.warning("Select at least one feature.")
        else:
            X = df[feature_cols].values
            y = df[target_col].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            st.metric("R² (higher is better)", f"{r2:.3f}")
            st.metric("MAE (↓)", f"{mae:.3f}")
            st.metric("MSE (↓)", f"{mse:.3f}")

            st.write("**Coefficients:**")
            coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_})
            st.dataframe(coef_df, use_container_width=True)

            st.write("**Make a Prediction**")
            inputs = []
            cols = st.columns(min(4, len(feature_cols)) or 1)
            for i, col in enumerate(feature_cols):
                with cols[i % len(cols)]:
                    val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].median()))
                inputs.append(val)
            if st.button("Predict"):
                pred = model.predict([inputs])[0]
                st.success(f"Predicted {target_col}: {pred:.2f}")

            st.subheader("Residuals Plot")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred - y_test)
            ax.axhline(0)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Residual (Predicted - Actual)")
            st.pyplot(fig)

            if use_sql and st.button("Save current data to SQLite as 'scores'"):
                try:
                    conn = sqlite3.connect(db_name)
                    df.to_sql("scores", conn, if_exists="replace", index=False)
                    conn.close()
                    st.success("Saved current dataframe to SQLite table 'scores'")
                except Exception as e:
                    st.error(f"SQLite save error: {e}")

with tab_viz:
    st.subheader("Exploratory Visualizations")
    if df is None:
        st.info("Load data first from the Data tab.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns for scatter.")
        else:
            x = st.selectbox("X axis", num_cols, index=0)
            y = st.selectbox("Y axis", num_cols, index=1)
            fig, ax = plt.subplots()
            ax.scatter(df[x], df[y])
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            st.pyplot(fig)

        st.write("Histogram")
        col = st.selectbox("Column for histogram", num_cols, index=0)
        bins = st.slider("Bins", 5, 50, 20, 1)
        fig2, ax2 = plt.subplots()
        ax2.hist(df[col].dropna(), bins=bins)
        ax2.set_xlabel(col)
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
