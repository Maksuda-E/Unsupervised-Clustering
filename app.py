#  imports streamlit for the web app
import streamlit as st

#  imports the prediction function
from src.predict import predict_cluster

#  sets the page configuration
st.set_page_config(
    page_title="Mall Customer Segmentation",
    layout="wide"
)

#  adds custom CSS styling for a new design and color theme
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

#  displays the main title
st.markdown(
    '<div class="main-title">Mall Customer Segmentation App</div>',
    unsafe_allow_html=True
)

#  displays the subtitle
st.markdown(
    '<div class="sub-title">Enter customer details to predict the customer segment</div>',
    unsafe_allow_html=True
)

#  defines six cluster meanings for the notebook based 3 feature flow
cluster_labels = {
    0: "Balanced Income and Spending Customers",
    1: "High Income and Low Spending Customers",
    2: "High Income and High Spending Customers",
    3: "Young Lower Income and High Spending Customers",
    4: "Young Lower Income and Low Spending Customers",
    5: "Older Moderate Income Customers"
}

#  creates a centered page layout
left_space, center_col, right_space = st.columns([0.6, 4, 0.6])

# This block contains the main layout
with center_col:

    #  creates two columns for the form and side panel
    form_col, side_col = st.columns([2.1, 1], gap="large")

    # This block creates the form section
    with form_col:

        #  displays the section title
        st.markdown('<div class="section-title">Customer Details</div>', unsafe_allow_html=True)

        #  creates two columns for the first row of inputs
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

        #  creates the spending score slider
        spending_score = st.slider(
            "Spending Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

        #  adds a little spacing before the button
        st.markdown("<br>", unsafe_allow_html=True)

        #  creates the prediction button
        predict_button = st.button("Predict Customer Segment")

    # This block creates the side overview section
    with side_col:

        #  displays the overview card
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-title">Customer Overview</div>
                <div class="summary-text">
                    This customer is {age} years old, has an annual income of {annual_income:.0f}, and a spending score of {spending_score:.0f}.
                    Based on the notebook flow, these are the three features used by the clustering model to assign the customer to one of six groups.
                    The segment helps describe similar shopping behavior patterns for targeting and analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        #  checks whether a previous prediction exists
        if "mall_cluster_result" in st.session_state:

            #  gets the stored cluster result
            result = st.session_state["mall_cluster_result"]

            #  gets the readable cluster meaning
            cluster_meaning = cluster_labels.get(result, "Unknown Customer Segment")

            #  displays the prediction result card
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

    #  checks whether the prediction button has been clicked
    if predict_button:

        #  prepares the model input dictionary
        user_input = {
            "Age": age,
            "Annual_Income": annual_income,
            "Spending_Score": spending_score
        }

        #  starts the safe prediction block
        try:
            #  gets the predicted cluster from the saved model
            result = predict_cluster(user_input)

            #  stores the result in session state
            st.session_state["mall_cluster_result"] = result

            #  reruns the app so the result appears immediately
            st.rerun()

        # This block handles prediction errors
        except Exception as exc:
            #  displays the error message
            st.error(f"Prediction failed: {exc}")

#  shows a short explanation under the main layout
st.info(
    "This segmentation app follows the notebook flow that uses Age, Annual Income, and Spending Score for clustering."
)
