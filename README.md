# Mall Customer Segmentation Project

## Overview

This project converts the Unsupervised Clustering notebook into a modular Python project and an interactive Streamlit web application.

The goal is to segment mall customers into meaningful groups using KMeans clustering based on their purchasing behavior.

The project follows the notebook workflow:
1. Explore clustering using Annual Income and Spending Score
2. Use Elbow Method (WCSS) to determine optimal clusters
3. Use Silhouette Score to validate cluster quality
4. Build final model using Age, Annual Income, and Spending Score
5. Final optimal number of clusters selected as 6


## Problem Statement

Mall businesses want to understand customer behavior to improve marketing strategies.

Using clustering, customers are grouped into segments based on:
- Income level
- Spending behavior
- Age


## Dataset

The dataset contains the following features:

- Customer_ID: Unique identifier
- Gender: Male or Female
- Age: Customer age
- Annual_Income: Income in thousands
- Spending_Score: Score between 1–100

Dataset file:data/mall_customers.csv


# Model Workflow
Load dataset
Select features: Age, Annual Income, Spending Score
Scale data using StandardScaler
Train KMeans clustering model
Evaluate clusters using:
Elbow Method (WCSS)
Silhouette Score
Select optimal clusters (k = 6)
Save model and artifacts
Use model in Streamlit app for predictions


# Cluster Interpretation (6 Clusters)

The final model segments customers into 6 groups:

High income, high spending customers
High income, low spending customers
Low income, high spending customers
Low income, low spending customers
Young average customers
Older average customers

These segments help businesses target marketing strategies effectively.

How to Run
# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python main.py

# Run Streamlit app
streamlit run app.py

# Streamlit Link
https://unsupervised-clustering-maksuda.streamlit.app/

# Deployment

The application is deployed using Streamlit Cloud.

Users can:

Input customer details
Predict customer segment
View cluster visualization
Understand customer group behavior

# Key Features
Modular code structure
Logging and exception handling
Model evaluation using proper clustering metrics
Interactive web application
cluster prediction

## Project Structure

```text
Unsupervised-Clustering/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── mall_customers.csv
│
├── artifacts/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── train_clustered_data.csv
│   ├── cluster_profiles.csv
│   ├── segment_profiles.json
│   ├── elbow_curve.csv
│   └── silhouette_scores.csv
│
├── logs/
│   └── project.log
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── custom_exception.py
    ├── data_loader.py
    ├── evaluate.py
    ├── logger.py
    ├── preprocessing.py
    ├── predict.py
    ├── train.py
    └── utils.py

