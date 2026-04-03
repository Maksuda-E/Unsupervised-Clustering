# Import json for reading saved segment meanings.
import json

# Import os for checking file existence.
import os

# Import pandas for reading csv data.
import pandas as pd

# Import plotly express for charts.
import plotly.express as px

# Import streamlit for the web app.
import streamlit as st

# Import app and project config classes.
from src.config import AppConfig, ArtifactConfig, DataConfig, ModelConfig

# Import predictor class.
from src.predict import ClusterPredictor

# Import training pipeline.
from src.train import run_training_pipeline


# Set page config.
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Add page styling.
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* REMOVE HEADER */
        header {visibility: hidden;}

        /* REMOVE TOP SPACE */
        .block-container {
            padding-top: 1rem !important;
        }

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


# Check whether required saved files already exist.
def artifacts_exist() -> bool:
    return (
        os.path.exists(ArtifactConfig.MODEL_PATH)
        and os.path.exists(ArtifactConfig.SCALER_PATH)
        and os.path.exists(ArtifactConfig.SEGMENT_PROFILE_PATH)
    )


# Train the model automatically if artifacts are missing.
def ensure_artifacts() -> None:
    if not artifacts_exist():
        run_training_pipeline()


# Load segment profile json file.
def load_segment_profiles() -> dict:
    if os.path.exists(AppConfig.SEGMENT_PROFILE_PATH):
        with open(AppConfig.SEGMENT_PROFILE_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


# Cache original data.
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DataConfig.DATA_PATH)


# Cache predictor object.
@st.cache_resource
def load_predictor() -> ClusterPredictor:
    ensure_artifacts()
    return ClusterPredictor()


# Build summary table.
def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if "Cluster" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("Cluster")[["Annual_Income", "Spending_Score", "Age"]]
        .mean()
        .round(2)
        .reset_index()
    )
    summary["Customer_Count"] = df["Cluster"].value_counts().sort_index().values
    return summary


# Main app function.
def main() -> None:
    inject_css()

    st.markdown(
        """
        <div class="hero-card">
            <h1>Mall Customer Segmentation App</h1>
            <p>
                This app predicts the customer segment using Age, Annual Income, and Spending Score.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictor = load_predictor()
    df = load_data()
    segment_profiles = load_segment_profiles()

    st.sidebar.header("Customer Input")

    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.slider("Age", min_value=18, max_value=70, value=30)
    annual_income = st.sidebar.slider("Annual Income", min_value=15, max_value=137, value=60)
    spending_score = st.sidebar.slider("Spending Score", min_value=1, max_value=99, value=50)

    predict_button = st.sidebar.button("Predict Cluster", use_container_width=True)

    col1, col2, col3 = st.columns(3)

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

    if predict_button:
        cluster_id = predictor.predict_single(
            age=age,
            annual_income=annual_income,
            spending_score=spending_score,
            gender=gender,
        )

        segment_name = segment_profiles.get(str(cluster_id), {}).get("segment_name", f"Cluster {cluster_id}")
        meaning = segment_profiles.get(str(cluster_id), {}).get("meaning", "No meaning available.")

        st.subheader("Prediction Result")
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

    st.subheader("Cluster Visualization")

    if os.path.exists(AppConfig.CLUSTERED_DATA_PATH):
        clustered_df = pd.read_csv(AppConfig.CLUSTERED_DATA_PATH)
    else:
        clustered_df = df.copy()

    if "Cluster" in clustered_df.columns:
        fig = px.scatter(
            clustered_df,
            x="Annual_Income",
            y="Spending_Score",
            color=clustered_df["Cluster"].astype(str),
            size="Age",
            hover_data=["Gender", "Age"],
            title="Annual Income and Spending Score by Cluster",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster Summary Table")
        summary_df = build_summary_table(clustered_df)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Cluster Meaning")

    seg_col1, seg_col2 = st.columns(2)
    sorted_keys = sorted(segment_profiles.keys(), key=lambda x: int(x))

    for index, key in enumerate(sorted_keys):
        profile = segment_profiles[key]
        target_col = seg_col1 if index % 2 == 0 else seg_col2

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


if __name__ == "__main__":
    main()
