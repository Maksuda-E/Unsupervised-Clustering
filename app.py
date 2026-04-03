# Import the json module to read saved segment descriptions.
import json

# Import the os module for file path checks.
import os

# Import pandas for reading clustered data if needed.
import pandas as pd

# Import plotly express for interactive charts in Streamlit.
import plotly.express as px

# Import streamlit to build the web app.
import streamlit as st

# Import project configuration classes.
from src.config import AppConfig, DataConfig, ModelConfig

# Import the predictor class used for single customer prediction.
from src.predict import ClusterPredictor


# Set the Streamlit page settings.
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Define a function to inject custom CSS styling into the app.
def inject_css() -> None:
    # Add custom styles for layout, cards, titles, and spacing.
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
        }

        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 18px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
            text-align: center;
        }

        .result-card {
            background: white;
            padding: 1.4rem;
            border-radius: 18px;
            border: 1px solid #dbeafe;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.10);
        }

        .segment-card {
            background: white;
            padding: 1.2rem;
            border-radius: 18px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
            min-height: 220px;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Define a function to load the saved segment profile JSON file.
def load_segment_profiles() -> dict:
    # Store the configured segment profile file path.
    profile_path = AppConfig.SEGMENT_PROFILE_PATH

    # Check whether the JSON file exists.
    if os.path.exists(profile_path):
        # Open the file in read mode with UTF 8 encoding.
        with open(profile_path, "r", encoding="utf-8") as file:
            # Return the parsed JSON content.
            return json.load(file)

    # Return an empty dictionary if the file is not found.
    return {}


# Cache the dataset so Streamlit does not reload it on every interaction.
@st.cache_data
def load_data() -> pd.DataFrame:
    # Read the original dataset CSV file.
    return pd.read_csv(DataConfig.DATA_PATH)


# Cache the predictor object for better app performance.
@st.cache_resource
def load_predictor() -> ClusterPredictor:
    # Create and return the predictor instance.
    return ClusterPredictor()


# Define a function to show cluster wise average values in table format.
def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    # Check if the Cluster column exists in the dataframe.
    if "Cluster" not in df.columns:
        # Return an empty dataframe if no cluster labels are present.
        return pd.DataFrame()

    # Group the data by Cluster and calculate mean feature values.
    summary = (
        df.groupby("Cluster")[["Annual_Income", "Spending_Score", "Age"]]
        .mean()
        .round(2)
        .reset_index()
    )

    # Add the number of customers in each cluster.
    summary["Customer_Count"] = df["Cluster"].value_counts().sort_index().values

    # Return the prepared summary table.
    return summary


# Define the main application function.
def main() -> None:
    # Inject the custom CSS into the page.
    inject_css()

    # Display the top hero section.
    st.markdown(
        """
        <div class="hero-card">
            <h1>Mall Customer Segmentation App</h1>
            <p>
                This app predicts the customer segment using Age, Annual Income, and Spending Score.
                The segment explanations are kept simple and based on the notebook's customer grouping logic.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load the predictor object.
    predictor = load_predictor()

    # Load the raw dataset.
    df = load_data()

    # Load the saved segment profile descriptions.
    segment_profiles = load_segment_profiles()

    # Create the sidebar title.
    st.sidebar.header("Customer Input")

    # Create a gender select box.
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

    # Create an age slider.
    age = st.sidebar.slider("Age", min_value=18, max_value=70, value=30)

    # Create an annual income slider.
    annual_income = st.sidebar.slider("Annual Income", min_value=15, max_value=137, value=60)

    # Create a spending score slider.
    spending_score = st.sidebar.slider("Spending Score", min_value=1, max_value=99, value=50)

    # Create a prediction button.
    predict_button = st.sidebar.button("Predict Cluster", use_container_width=True)

    # Create three columns for quick metrics.
    col1, col2, col3 = st.columns(3)

    # Show total records metric.
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{df.shape[0]}</h3>
                <p>Total Customers</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Show cluster count metric.
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{ModelConfig.N_CLUSTERS}</h3>
                <p>Cluster Groups</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Show feature count metric.
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{len(ModelConfig.FEATURE_COLUMNS)}</h3>
                <p>Model Features</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Run prediction only if the button is pressed.
    if predict_button:
        # Get the predicted cluster id from the predictor.
        cluster_id = predictor.predict_single(
            age=age,
            annual_income=annual_income,
            spending_score=spending_score,
            gender=gender,
        )

        # Get the saved segment name from the profile dictionary.
        segment_name = segment_profiles.get(str(cluster_id), {}).get("segment_name", f"Cluster {cluster_id}")

        # Get the saved simple meaning from the profile dictionary.
        meaning = segment_profiles.get(str(cluster_id), {}).get("meaning", "No meaning available.")

        # Show the prediction title.
        st.subheader("Prediction Result")

        # Display prediction result in a styled card.
        st.markdown(
            f"""
            <div class="result-card">
                <h2>Predicted Cluster: {cluster_id}</h2>
                <h3>{segment_name}</h3>
                <p><strong>Simple Meaning:</strong> {meaning}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Show the chart section title.
    st.subheader("Cluster Visualization")

    # Check if clustered dataset exists.
    clustered_file = AppConfig.CLUSTERED_DATA_PATH

    # Load clustered data if available, otherwise use raw data.
    if os.path.exists(clustered_file):
        # Read the clustered dataset from artifacts.
        clustered_df = pd.read_csv(clustered_file)
    else:
        # Use original raw dataset if clustered file does not exist.
        clustered_df = df.copy()

    # Only create chart if cluster labels are available.
    if "Cluster" in clustered_df.columns:
        # Build the scatter chart from notebook style logic.
        fig = px.scatter(
            clustered_df,
            x="Annual_Income",
            y="Spending_Score",
            color=clustered_df["Cluster"].astype(str),
            size="Age",
            hover_data=["Gender", "Age"],
            title="Annual Income and Spending Score by Cluster",
        )

        # Update chart height.
        fig.update_layout(height=520)

        # Show the chart in the app.
        st.plotly_chart(fig, use_container_width=True)

        # Show the summary table.
        st.subheader("Cluster Summary Table")

        # Build the summary table.
        summary_df = build_summary_table(clustered_df)

        # Display the summary table.
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Show the simple segment explanation section.
    st.subheader("Cluster Meaning")

    # Create columns for segment cards.
    seg_col1, seg_col2 = st.columns(2)

    # Get the sorted cluster keys from the saved profiles.
    sorted_keys = sorted(segment_profiles.keys(), key=lambda x: int(x))

    # Loop through every cluster profile.
    for index, key in enumerate(sorted_keys):
        # Get the current profile.
        profile = segment_profiles[key]

        # Alternate the card placement between two columns.
        target_col = seg_col1 if index % 2 == 0 else seg_col2

        # Show the card inside the selected column.
        with target_col:
            st.markdown(
                f"""
                <div class="segment-card">
                    <h3>Cluster {key}</h3>
                    <h4>{profile.get("segment_name", "")}</h4>
                    <p>{profile.get("meaning", "")}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Show the notebook logic note.
    st.subheader("Notebook Logic Used")

    # Display the logic in plain language.
    st.write(
        "The notebook first explored 5 customer groups using Annual Income and Spending Score. "
        "Those 5 group meanings are used here because they are the clearest and easiest to explain."
    )


# Run the main function only when the file is executed directly.
if __name__ == "__main__":
    main()