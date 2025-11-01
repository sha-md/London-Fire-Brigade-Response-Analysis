# 🚒 London Fire Brigade – Response Time Analysis & Prediction

 A data science project analyzing and predicting **London Fire Brigade (LFB)** response times using machine learning.  
Originally developed as a **group project** at *DataScientest (Germany)*, later extended into an **interactive Streamlit web app** by **Shabnam B. Mahammad**.

---

##  Table of Contents
- [Project Background](#project-background)
- [Business Objective](#business-objective)
- [Streamlit Web App](#streamlit-web-app)
- [Key Features](#key-features)
- [Main Insights](#main-insights)
- [Machine Learning Overview](#machine-learning-overview)
- [Learnings](#learnings)
- [Repository Structure](#repository-structure)
- [Author](#author)

---

##  Project Background

This project explores how fast the **London Fire Brigade** responds to incidents across different boroughs between **2009–2023**.  
It applies **data cleaning, feature engineering, and ML modeling** to estimate and visualize emergency response times — and even forecast trends up to **2030**.

---

##  Business Objective

The goal of this project is to help the **London Fire Brigade (LFB)** gain actionable insights into its emergency response operations and identify factors causing delays.  

By analyzing historical incident and mobilisation data, the project aims to:  
- Understand how **response times vary** by borough, time of day, and incident type  
- Predict **expected arrival times** for new emergency calls  
- Forecast **future trends** in response performance (up to 2030)  
- Support **strategic resource planning** and station-level optimization  

From a business standpoint, improving response time not only enhances **public safety** and **operational efficiency**, but also reduces **property damage** and strengthens community trust in emergency services.

--

##  Streamlit Web App

🔗 **[Launch the App](https://sha-md-london-fire-brigade-response-analysis-app-ru7by8.streamlit.app/))**  

---

##  Key Features

-  **Dashboard:** Yearly, hourly, and borough-wise response metrics  
-  **Predictor:** ML-based estimation of response time by hour, year & borough  
-  **Forecast:** Future response trend (2024–2030) using model simulation  
-  **Fast Performance:** Cached model & datasets for instant predictions  
-  **Model:** RandomForestRegressor pipeline with OneHotEncoding  

---

##  Main Insights

- Average response time: **≈ 6 minutes**  
- Borough & distance are key predictors of delay  
- Historical trend shows **steady improvement** over the years  
- Model forecast suggests further **optimization by 2030**

---

##  Machine Learning Overview

| Task | Model Used | Metric |
|------|-------------|---------|
| Regression (Response Time) | XGBoost, RandomForest | MAE ≈ 56 sec |
| Deployment | Streamlit + Cached RandomForest | Fast inference |
| Forecast | Synthetic simulation (2009–2030) | Trend visualization |

---

##  Learnings

- Applied **end-to-end data science pipeline** on real-world public data  
- Improved **large dataset handling** (2M+ rows)  
- Deployed an **ML-powered Streamlit app** for public visualization  
- Strengthened skills in **feature engineering, model tuning, and caching**

---

##  Author

**Author:** [SHABNAM B MAHAMMAD]  
 shabnam71.md@gmail.com
🔗 [LinkedIn](www.linkedin.com/in/shabnam-b-mahammad-377520272) 

> “Analyzing the past to predict the response — one dataset at a time.” 🚒
