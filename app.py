import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
import seaborn as sns

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="London Fire Brigade ML Dashboard", page_icon="🚒", layout="wide")
st.title("🚒 London Fire Brigade Response Time Analysis")
st.markdown("""
Upload a dataset and build regression or classification models using **XGBoost** or **LightGBM**.
""")

# ---------------- SIDEBAR ----------------
task_type = st.sidebar.selectbox("Select Task Type", ["Regression", "Classification"])
model_type = st.sidebar.selectbox("Select Model", ["XGBoost", "LightGBM"])
test_size_ratio = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
run_button = st.sidebar.button("🚀 Train Model")

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Select target column
    target_col = st.selectbox("Select Target Column", df.columns)

    if run_button:
        st.write("### Model Training and Evaluation")

        # Split data
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle categorical encoding
        X = pd.get_dummies(X, drop_first=True)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)

        # Scale numeric columns
        num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        # Train models
        if task_type == "Regression":
            if model_type == "XGBoost":
                model = xgb.XGBRegressor(
                    n_estimators=300, learning_rate=0.05,
                    max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method='hist'
                )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.05,
                    max_depth=6, subsample=0.8, colsample_bytree=0.8
                )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"**Mean Absolute Error (MAE):** {mae:.3f}")
            st.write(f"**R² Score:** {r2:.3f}")

            # Actual vs Predicted plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.4)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{model_type} Regression: Actual vs Predicted")
            st.pyplot(fig)

        else:  # Classification
            if model_type == "XGBoost":
                model = xgb.XGBClassifier(
                    n_estimators=300, learning_rate=0.1,
                    max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method='hist'
                )
            else:
                model = lgb.LGBMClassifier(
                    n_estimators=300, learning_rate=0.1,
                    max_depth=6, subsample=0.8, colsample_bytree=0.8
                )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {acc:.3f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        # Feature importance
        st.write("### Feature Importance")
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(importances)), importances[sorted_idx])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(np.array(X.columns)[sorted_idx], rotation=90)
        st.pyplot(fig)

        # SHAP explanation
        st.write("### 🔍 SHAP Summary Plot")
        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            st.warning(f"SHAP explanation skipped: {e}")
else:
    st.info("👆 Upload a dataset (CSV) to begin.")


