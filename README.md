# London Fire Brigade – Response Time Analysis & Prediction

A data science and machine learning project analyzing and predicting **London Fire Brigade (LFB)** response times.  
Developed as a **group project** at *DataScientest (Germany)* and later extended into an **interactive Streamlit web application** by **Shabnam B. Mahammad**.

---

## Table of Contents
- [Project Background](#project-background)
- [Business Objective](#business-objective)
- [Why This Project Matters](#why-this-project-matters)
- [Streamlit Web App](#streamlit-web-app)
- [Technical Workflow](#technical-workflow)
- [Model Interpretability](#model-interpretability)
- [Cost–Benefit Impact](#costbenefit-impact)
- [Key Results](#key-results)
- [Author](#author)

---

## Project Background

This project explores how quickly the **London Fire Brigade** responds to incidents across different boroughs between 2009 and 2023.  
It applies data cleaning, feature engineering, and machine learning modeling to estimate and visualize emergency response times — and forecast future performance up to 2030.

---

## Business Objective

The project aims to provide the London Fire Brigade (LFB) with data-driven insights to:

1. Analyze how response times vary by borough, time of day, and incident type.  
2. Identify operational and geographic factors contributing to longer response times.  
3. Predict expected arrival times for new incidents.  
4. Forecast future performance and support strategic station planning.

Improving response times enhances **public safety**, **reduces property damage**, and optimizes **operational efficiency** within the organization.

---

## Why This Project Matters

Predicting emergency response times is not just a data exercise — it has real-world impact.

- Faster emergency responses can save lives and reduce property losses.  
- Insights can improve crew allocation, resource planning, and station placement.  
- Predictive models enable proactive, evidence-based decision-making.  

A 10-second improvement in the citywide average response time can impact thousands of emergency outcomes annually.

---

## Streamlit Web App

Live Demo: **[Click Here to Open](https://sha-md-london-fire-brigade-response-analysis-app-ru7by8.streamlit.app/)**

This app allows users to:
- Visualize historical response data by borough and year.  
- Predict response times based on input features (borough, hour, year).  
- Forecast trends for 2024–2030 using trained machine learning models.  
- Explore data interactively in a simple, user-friendly interface.

Technologies used: **Streamlit, Python, Pandas, Plotly, scikit-learn**

---

## Technical Workflow

1. **Data Sources**
   - London Fire Brigade *Incident* and *Mobilisation* datasets (2009–2023).  
   - Enriched with geographic coordinates for spatial analysis.  

2. **Data Cleaning and Feature Engineering**
   - Removed anomalies (pre-2014 data, 2020 COVID impact).  
   - Extracted new columns: `TravelDistance`, `TravelSpeed`, `Station_Loc_same_incident_Loc`.  
   - Extracted temporal features: `Year`, `Month`, `Weekday`, `Hour`.  

3. **Model Development**
   - Tested Linear Regression, Decision Tree, Random Forest, and XGBoost.  
   - Selected **XGBoost** for best accuracy and **Random Forest** for deployment.  
   - Final model Mean Absolute Error (MAE): **56 seconds**.  

4. **Deployment**
   - Built and deployed using **Streamlit Cloud** with cached model inference for fast performance.  
   - Data hosted through GitHub Releases (.csv.gz format).  

---

## Model Interpretability

Explainability was achieved using **SHAP (SHapley Additive exPlanations)** to understand which features most influence predictions.

| Feature | Influence | Description |
|----------|------------|-------------|
| TravelDistance | High | Longer distances increase response time. |
| Borough | Medium | Denser boroughs slightly slower. |
| HourOfCall | Medium | Rush hours cause longer travel. |
| Station_Loc_same_incident_Loc | Low | If true, responses are faster. |
| CalYear | Negative | Consistent improvement over time. |

Understanding *why* response times differ helps optimize station locations, traffic routing, and resource allocation.

---

## Cost–Benefit Impact

- Reducing the mean response time by **1 minute** could save hundreds of thousands of pounds annually in reduced fire damage and insurance payouts.  
- Operational efficiency improvements could lead to **5–8% cost reduction** in staff and vehicle utilization.  
- Predictive insights enable budget optimization while maintaining high service quality.

This shows how machine learning can generate **economic as well as operational value** for public safety organizations.

---

## Key Results

| Metric | Model | Value |
|---------|--------|--------|
| Mean Absolute Error (MAE) | XGBoost | 56 seconds |
| Average Response Time | Historical (2009–2023) | ≈ 6 minutes |
| Forecast Trend | 2024–2030 | Gradual improvement (−4%) |

**Business Insight:**  
High-travel-distance boroughs can improve outcomes through better station allocation and route optimization.

---

## Author

**Shabnam Begam Mahammad**  
[LinkedIn](https://www.linkedin.com/in/shabnam-b-mahammad) | 
[Email](mailto:shabnam71.md@gmail.com) | 

