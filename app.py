# This line imports json for loading cluster mapping
import json

# This line imports os for file path checking
import os

# This line imports streamlit for the web app
import streamlit as st

# This line imports the prediction function
from src.predict import predict_cluster

# This line sets the page configuration
st.set_page_config(
    page_title="Mall Customer Segmentation",
    layout="wide"
)

# This line defines the cluster mapping file path
CLUSTER_MAPPING_FILE_PATH = "artifacts/cluster_mapping.json"

# This function loads saved cluster mapping if available
def load_cluster_mapping():
    # This line checks whether the mapping file exists
    if os.path.exists(CLUSTER_MAPPING_FILE_PATH):
        # This line opens the mapping file
        with open(CLUSTER_MAPPING_FILE_PATH, "r", encoding="utf-8") as file:
            # This line loads the mapping data
            mapping = json.load(file)

        # This line converts mapping keys back to integers
        return {int(key): value for key, value in mapping.items()}

    # This line returns default mapping if file does not exist
    return {
        0: "Older moderate customers.",
        1: "Balanced customers.",
        2: "High income low spending customers.",
        3: "High income high spending customers.",
        4: "Low income high spending customers.",
        5: "Low income low spending customers."
    }

# This line adds custom CSS styling for the design and cards
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fdf4ff, #f5f3ff, #eef2ff);
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        color: #3b0764;
        margin-bottom: 0.25rem;
    }

    .sub-title {
        font-size: 1.05rem;
        text-align: center;
        color: #5b5870;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #4c1d95;
        margin-bottom: 1rem;
    }

    .summary-card {
        background: linear-gradient(135deg, #7c3aed, #db2777);
        color: white;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 14px 30px rgba(124, 58, 237, 0.22);
        margin-bottom: 1rem;
    }

    .summary-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .summary-text {
        font-size: 1rem;
        line-height: 1.75;
        opacity: 0.97;
    }

    .result-card {
        background: linear-gradient(135deg, #0f766e, #2563eb);
        color: white;
        border-radius: 20px;
        padding: 20px;
        margin-top: 1rem;
        box-shadow: 0 12px 26px rgba(37, 99, 235, 0.20);
    }

    .result-title {
        font-size: 0.95rem;
        margin-bottom: 0.45rem;
        opacity: 0.95;
    }

    .result-value {
        font-size: 1.65rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .result-text {
        font-size: 1rem;
        opacity: 0.96;
    }

    .guide-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(196, 181, 253, 0.50);
        border-radius: 22px;
        padding: 24px;
        margin-top: 1.5rem;
        box-shadow: 0 12px 28px rgba(76, 29, 149, 0.10);
    }

    .guide-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #312e81;
        margin-bottom: 1.2rem;
    }

    .guide-item {
        margin-bottom: 1rem;
        color: #374151;
        line-height: 1.7;
    }

    .guide-item b {
        color: #4c1d95;
    }

    .guide-note {
        margin-top: 1rem;
        color: #4b5563;
        font-size: 0.98rem;
        line-height: 1.7;
    }

    div.stButton > button {
        width: 100%;
        height: 50px;
        border: none;
        border-radius: 14px;
        background: linear-gradient(90deg, #9333ea, #ec4899);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 10px 22px rgba(147, 51, 234, 0.20);
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #7e22ce, #db2777);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# This line loads the cluster labels from saved mapping
cluster_labels = load_cluster_mapping()

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

# This line creates a centered page layout
left_space, center_col, right_space = st.columns([0.6, 4, 0.6])

# This block contains the main layout
with center_col:

    # This line creates two columns for the form and side panel
    form_col, side_col = st.columns([2.1, 1], gap="large")

    # This block creates the form section
    with form_col:

        # This line displays the section title
        st.markdown('<div class="section-title">Customer Details</div>', unsafe_allow_html=True)

        # This line creates two columns for age and annual income
        row1_col1, row1_col2 = st.columns(2)

        # This block creates the age input
        with row1_col1:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=100,
                value=25,
                step=1
            )

        # This block creates the annual income input
        with row1_col2:
            annual_income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=60.0,
                step=1.0
            )

        # This line creates the spending score slider
        spending_score = st.slider(
            "Spending Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

        # This line adds spacing before the button
        st.markdown("<br>", unsafe_allow_html=True)

        # This line creates the prediction button
        predict_button = st.button("Predict Customer Segment")

    # This block creates the side overview section
    with side_col:

        # This line displays the overview card
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-title">Customer Overview</div>
                <div class="summary-text">
                    This customer is {age} years old, has an annual income of {annual_income:.0f}, and a spending score of {spending_score:.0f}.
                    These are the three features used by the clustering model to assign the customer to one of six clusters.
                    The final cluster number is generated by the trained KMeans model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # This line checks whether a previous prediction exists
        if "mall_cluster_result" in st.session_state:

            # This line gets the stored cluster result
            result = st.session_state["mall_cluster_result"]

            # This line gets the mapped cluster meaning
            cluster_meaning = cluster_labels.get(result, f"Customer behavior pattern for cluster {result}.")

            # This line displays the prediction result card
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Predicted Segment</div>
                    <div class="result-value">Cluster {result}</div>
                    <div class="result-text">{cluster_meaning}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # This line checks whether the prediction button has been clicked
    if predict_button:

        # This line prepares the model input dictionary
        user_input = {
            "Age": age,
            "Annual_Income": annual_income,
            "Spending_Score": spending_score
        }

        # This line starts the safe prediction block
        try:
            # This line gets the predicted cluster
            result = predict_cluster(user_input)

            # This line stores the result in session state
            st.session_state["mall_cluster_result"] = result

            # This line reruns the app so the result appears immediately
            st.rerun()

        # This block handles prediction errors
        except Exception as exc:
            # This line displays the error message
            st.error(f"Prediction failed: {exc}")

# This line shows the cluster meaning guide as a styled card
guide_html = f"""
<div class="guide-card">
    <div class="guide-title">Segment Meaning Guide</div>

    <div class="guide-item">
        <b>Cluster 0</b><br>
        {cluster_labels.get(0, "Customer behavior pattern for cluster 0.")}
    </div>

    <div class="guide-item">
        <b>Cluster 1</b><br>
        {cluster_labels.get(1, "Customer behavior pattern for cluster 1.")}
    </div>

    <div class="guide-item">
        <b>Cluster 2</b><br>
        {cluster_labels.get(2, "Customer behavior pattern for cluster 2.")}
    </div>

    <div class="guide-item">
        <b>Cluster 3</b><br>
        {cluster_labels.get(3, "Customer behavior pattern for cluster 3.")}
    </div>

    <div class="guide-item">
        <b>Cluster 4</b><br>
        {cluster_labels.get(4, "Customer behavior pattern for cluster 4.")}
    </div>

    <div class="guide-item">
        <b>Cluster 5</b><br>
        {cluster_labels.get(5, "Customer behavior pattern for cluster 5.")}
    </div>

    <div class="guide-note">
        <b>Note:</b> These descriptions are general cluster interpretations based on the trained clustering model.
    </div>
</div>
"""

# This line renders the guide card
st.markdown(guide_html, unsafe_allow_html=True)