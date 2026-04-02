# This line imports streamlit for the web app
import streamlit as st

# This line imports the prediction function
from src.predict import predict_cluster

# This line imports json for reading saved summary files if available
import json

# This line imports os for checking file paths
import os

# This line sets the page configuration
st.set_page_config(
    page_title="Mall Customer Segmentation",
    layout="wide"
)

# This line adds custom CSS styling for the design
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fff7ed, #fffbeb, #f0fdfa);
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        text-align: center;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1.05rem;
        text-align: center;
        color: #4b5563;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1rem;
    }

    .segment-card {
        background: linear-gradient(135deg, #f59e0b, #14b8a6);
        color: white;
        padding: 18px;
        border-radius: 18px;
        margin-top: 16px;
    }

    .segment-value {
        font-size: 1.5rem;
        font-weight: 800;
    }

    div.stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 14px;
        border: none;
        background: linear-gradient(90deg, #f59e0b, #0f766e);
        color: white;
        font-size: 1rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# This line displays the title
st.markdown('<div class="main-title">Mall Customer Segmentation App</div>', unsafe_allow_html=True)

# This line displays the subtitle
st.markdown(
    '<div class="sub-title">Enter customer details to predict the customer segment</div>',
    unsafe_allow_html=True
)

# This line defines cluster meanings
cluster_labels = {
    0: "Moderate Income and Low Spending Customers",
    1: "Moderate Income and High Spending Customers",
    2: "High Income and High Spending Customers",
    3: "Low Income and High Spending Customers",
    4: "High Income and Low Spending Customers"
}

# This line creates layout columns
left_col, _ = st.columns([2, 1])

# This block handles input section
with left_col:

    # This line shows section title
    st.markdown('<div class="section-title">Customer Details</div>', unsafe_allow_html=True)

    # This line creates inner columns
    col1, col2 = st.columns(2)

    # First column inputs
    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=25,
            step=1
        )

        spending_score = st.slider(
            "Spending Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

    # Second column inputs
    with col2:
        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=60.0,
            step=1.0
        )

    # This line creates predict button
    if st.button("Predict Customer Segment"):

        # This line prepares input data
        user_input = {
            "Age": age,
            "Annual_Income": annual_income,
            "Spending_Score": spending_score
        }

        try:
            # This line gets prediction
            result = predict_cluster(user_input)

            # This line gets cluster meaning
            cluster_meaning = cluster_labels.get(result, "Unknown Customer Segment")

            # This line displays result
            st.markdown(
                f'''
                <div class="segment-card">
                    <div class="segment-value">Cluster {result}</div>
                    <div>{cluster_meaning}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

            # This line shows explanation
            st.info(
                "Clusters represent groups of customers with similar behavior."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
