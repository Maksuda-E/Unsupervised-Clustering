# This line imports streamlit for building the web application
import streamlit as st

# This line imports the prediction function
from src.predict import predict_cluster

# This line imports json for loading saved metrics if available
import json

# This line imports os for checking whether files exist
import os

# This line sets the page configuration
st.set_page_config(
    page_title="Mall Customer Segmentation",
    layout="wide"
)

# This line adds custom CSS for styling the whole application
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc, #ecfeff, #fefce8);
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        text-align: center;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1.05rem;
        text-align: center;
        color: #475569;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
    }

    .summary-box {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }

    .result-box {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        padding: 20px;
        border-radius: 18px;
        margin-top: 16px;
        box-shadow: 0 10px 24px rgba(99, 102, 241, 0.22);
    }

    .result-title {
        font-size: 0.95rem;
        margin-bottom: 0.35rem;
        opacity: 0.95;
    }

    .result-value {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .result-text {
        font-size: 1rem;
    }

    div.stButton > button {
        width: 100%;
        height: 48px;
        border: none;
        border-radius: 14px;
        background: linear-gradient(90deg, #0284c7, #4f46e5);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(79, 70, 229, 0.20);
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #0369a1, #4338ca);
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.70);
        border: 1px solid rgba(203, 213, 225, 0.7);
        border-radius: 16px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# This line displays the main title
st.markdown(
    '<div class="main-title">Mall Customer Segmentation App</div>',
    unsafe_allow_html=True
)

# This line displays the subtitle
st.markdown(
    '<div class="sub-title">Enter customer details to predict the customer segment</div>',
    unsafe_allow_html=True
)

# This line creates a dictionary that explains each cluster label
cluster_labels = {
    0: "Moderate Income and Low Spending Customers",
    1: "Moderate Income and High Spending Customers",
    2: "High Income and High Spending Customers",
    3: "Low Income and High Spending Customers",
    4: "High Income and Low Spending Customers"
}

# This line creates two main columns for the page layout
left_col, right_col = st.columns([2, 1], gap="large")

# This block creates the left side input area
with left_col:

    # This line displays the customer details section title
    st.markdown('<div class="section-title">Customer Details</div>', unsafe_allow_html=True)

    # This line creates two inner columns for input alignment
    col1, col2 = st.columns(2)

    # This block contains the left inner column inputs
    with col1:
        # This line creates the age input
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=25,
            step=1
        )

        # This line creates the spending score slider
        spending_score = st.slider(
            "Spending Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

    # This block contains the right inner column inputs
    with col2:
        # This line creates the annual income input
        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=60.0,
            step=1.0
        )

    # This line adds a little space before the button
    st.markdown("<br>", unsafe_allow_html=True)

    # This line creates the prediction button
    predict_button = st.button("Predict Customer Segment")

    # This line checks whether the prediction button was clicked
    if predict_button:
        # This line creates the user input dictionary for the model
        user_input = {
            "Age": age,
            "Annual_Income": annual_income,
            "Spending_Score": spending_score
        }

        # This line starts the prediction block
        try:
            # This line gets the cluster prediction from the model
            result = predict_cluster(user_input)

            # This line gets the human readable label for the cluster
            cluster_meaning = cluster_labels.get(result, "Unknown Customer Segment")

            # This line displays the prediction result in a styled box
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-title">Predicted Customer Segment</div>
                    <div class="result-value">Cluster {result}</div>
                    <div class="result-text">{cluster_meaning}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # This line displays an explanation message
            st.info(
                "Cluster labels are identifiers created by KMeans. "
                "Customers in the same cluster usually show similar behavior."
            )

        # This block handles errors during prediction
        except Exception as exc:
            # This line displays the prediction error
            st.error(f"Prediction failed: {exc}")

# This block creates the right side summary area
with right_col:

    # This line displays the model summary section title
    st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)

    # This line opens a styled summary container
    st.markdown('<div class="summary-box">', unsafe_allow_html=True)

    # This line displays the model type label
    st.write("Model Type")

    # This line displays the model type value
    st.write("KMeans Clustering")

    # This line displays the features heading
    st.write("Input Features")

    # This line displays the input features used by the model
    st.write("Age, Annual Income, Spending Score")

    # This line closes the first summary container
    st.markdown('</div>', unsafe_allow_html=True)

    # This line starts a try block for loading saved metrics
    try:
        # This line sets the path to the metrics file
        metrics_path = "artifacts/metrics.json"

        # This line checks whether the metrics file exists
        if os.path.exists(metrics_path):

            # This line opens and loads the metrics file
            with open(metrics_path, "r") as file:
                metrics = json.load(file)

            # This line displays the saved metrics heading
            st.markdown('<div class="section-title">Saved Metrics</div>', unsafe_allow_html=True)

            # This line loops through the metrics and displays them one by one
            for key, value in metrics.items():
                try:
                    st.metric(label=key, value=round(float(value), 4))
                except Exception:
                    st.metric(label=key, value=value)

        # This block runs if the metrics file does not exist
        else:
            # This line displays an info message
            st.info("No saved summary file found. Train the model first if you want metrics here.")

    # This block handles metric loading errors
    except Exception as exc:
        # This line displays the metric loading error
        st.error(f"Could not load summary: {exc}")

    # This line displays the segment guide heading
    st.markdown('<div class="section-title">Segment Guide</div>', unsafe_allow_html=True)

    # This line opens another styled summary container
    st.markdown('<div class="summary-box">', unsafe_allow_html=True)

    # This line displays the first segment description
    st.caption("Cluster 0: Moderate income and low spending")

    # This line displays the second segment description
    st.caption("Cluster 1: Moderate income and high spending")

    # This line displays the third segment description
    st.caption("Cluster 2: High income and high spending")

    # This line displays the fourth segment description
    st.caption("Cluster 3: Low income and high spending")

    # This line displays the fifth segment description
    st.caption("Cluster 4: High income and low spending")

    # This line closes the final summary container
    st.markdown('</div>', unsafe_allow_html=True)
