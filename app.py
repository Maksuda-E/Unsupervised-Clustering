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

# This line adds custom CSS styling for a new design and color theme
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

    .panel {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(251, 191, 36, 0.20);
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(31, 41, 55, 0.08);
        backdrop-filter: blur(10px);
    }

    .panel-title {
        font-size: 1.2rem;
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
        box-shadow: 0 12px 24px rgba(20, 184, 166, 0.18);
    }

    .segment-heading {
        font-size: 0.95rem;
        opacity: 0.95;
        margin-bottom: 0.35rem;
    }

    .segment-value {
        font-size: 1.5rem;
        font-weight: 800;
    }

    .segment-text {
        margin-top: 8px;
        font-size: 1rem;
    }

    div.stButton > button {
        width: 100%;
        height: 48px;
        border: none;
        border-radius: 14px;
        background: linear-gradient(90deg, #f59e0b, #0f766e);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(245, 158, 11, 0.20);
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #d97706, #115e59);
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.70);
        border: 1px solid rgba(209, 213, 219, 0.6);
        border-radius: 16px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# This line displays the main title of the page
st.markdown('<div class="main-title">Mall Customer Segmentation App</div>', unsafe_allow_html=True)

# This line displays the subtitle below the title
st.markdown(
    '<div class="sub-title">Enter customer details to predict the customer segment</div>',
    unsafe_allow_html=True
)

# This line creates the cluster label dictionary
cluster_labels = {
    0: "Moderate Income and Low Spending Customers",
    1: "Moderate Income and High Spending Customers",
    2: "High Income and High Spending Customers",
    3: "Low Income and High Spending Customers",
    4: "High Income and Low Spending Customers"
}

# This line creates two main columns for layout
left_col, right_col = st.columns([2, 1])

# This block creates the left input section
with left_col:

    # This line starts a styled panel
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # This line shows the section title
    st.markdown('<div class="panel-title">Customer Details</div>', unsafe_allow_html=True)

    # This line creates two inner columns for better form layout
    col1, col2 = st.columns(2)

    # This block creates inputs in the first column
    with col1:
        # This line creates a numeric input for age
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=25,
            step=1
        )

        # This line creates a slider for spending score
        spending_score = st.slider(
            "Spending Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

    # This block creates inputs in the second column
    with col2:
        # This line creates a numeric input for annual income
        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=60.0,
            step=1.0
        )

    # This line creates the prediction button
    predict_button = st.button("Predict Customer Segment")

    # This line checks if the prediction button is clicked
    if predict_button:
        # This line creates the model input dictionary
        user_input = {
            "Age": age,
            "Annual_Income": annual_income,
            "Spending_Score": spending_score
        }

        # This line starts the prediction try block
        try:
            # This line gets the cluster prediction
            result = predict_cluster(user_input)

            # This line gets the human readable segment meaning
            cluster_meaning = cluster_labels.get(result, "Unknown Customer Segment")

            # This line shows the result in a custom result card
            st.markdown(
                f'''
                <div class="segment-card">
                    <div class="segment-heading">Predicted Cluster</div>
                    <div class="segment-value">Cluster {result}</div>
                    <div class="segment-text">{cluster_meaning}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

            # This line shows a helpful note below the result
            st.info(
                "Cluster labels are grouping identifiers from KMeans. "
                "Customers in the same cluster share similar purchasing behavior."
            )

        # This block handles prediction errors
        except Exception as exc:
            # This line shows the prediction error
            st.error(f"Prediction failed: {exc}")

    # This line closes the styled panel
    st.markdown('</div>', unsafe_allow_html=True)

# This block creates the right summary section
with right_col:

    # This line starts a styled summary panel
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # This line shows the summary title
    st.markdown('<div class="panel-title">Model Summary</div>', unsafe_allow_html=True)

    # This line shows static model information
    st.write("Model Type")
    st.write("KMeans Clustering")

    # This line shows the input features used by the model
    st.write("Input Features")
    st.write("Age, Annual Income, Spending Score")

    # This line starts a try block for loading optional metrics or config summary
    try:
        # This line sets a possible summary file path
        metrics_path = "artifacts/metrics.json"

        # This line checks whether the summary file exists
        if os.path.exists(metrics_path):
            # This line opens the summary file
            with open(metrics_path, "r") as file:
                metrics = json.load(file)

            # This line shows a heading for loaded metrics
            st.write("Saved Metrics")

            # This line loops through the metrics and displays them
            for key, value in metrics.items():
                try:
                    st.metric(label=key, value=round(float(value), 4))
                except Exception:
                    st.metric(label=key, value=value)
        else:
            # This line shows a message if no summary file is found
            st.info("No saved summary file found. Train the model first if you want metrics here.")

    # This block handles file loading errors
    except Exception as exc:
        # This line shows an error if summary loading fails
        st.error(f"Could not load summary: {exc}")

    # This line shows a quick segment guide heading
    st.write("Segment Guide")

    # This line shows a short guide for the segments
    st.caption("Cluster 0: Moderate income and low spending")
    st.caption("Cluster 1: Moderate income and high spending")
    st.caption("Cluster 2: High income and high spending")
    st.caption("Cluster 3: Low income and high spending")
    st.caption("Cluster 4: High income and low spending")

    # This line closes the styled summary panel
    st.markdown('</div>', unsafe_allow_html=True)
