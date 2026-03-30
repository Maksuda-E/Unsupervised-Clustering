# This line imports streamlit for the web app
import streamlit as st

# This line imports the prediction function
from src.predict import predict_cluster

# This line sets the page title and layout
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

# This line shows the app title
st.title("Mall Customer Segmentation App")

# This line shows a short description
st.write("Enter customer details to predict the customer segment.")

# This line creates a numeric input for age
age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)

# This line creates a numeric input for annual income
annual_income = st.number_input("Annual Income", min_value=0.0, value=60.0, step=1.0)

# This line creates a numeric input for spending score
spending_score = st.number_input("Spending Score", min_value=0.0, max_value=100.0, value=50.0, step=1.0)

# This line creates a dictionary to explain cluster meanings
cluster_labels = {
    0: "Moderate Income - Low Spending Customers",
    1: "Moderate Income - High Spending Customers",
    2: "High Income - High Spending Customers",
    3: "Low Income - High Spending Customers",
    4: "High Income - Low Spending Customers"
}

# This line checks if the Predict button is clicked
if st.button("Predict Cluster"):
    # This line creates the input dictionary
    user_input = {
        "Age": age,
        "Annual_Income": annual_income,
        "Spending_Score": spending_score
    }

    # This line starts a try block
    try:
        # This line gets the predicted cluster
        result = predict_cluster(user_input)

        # This line gets the cluster meaning from the dictionary
        cluster_meaning = cluster_labels.get(result, "Unknown Customer Segment")

        # This line displays the predicted cluster number
        st.success(f"Predicted Customer Cluster: {result}")

        # This line displays the cluster meaning
        st.info(f"Segment: {cluster_meaning}")

        # This line shows a note to explain cluster labels
        st.write("Note: Cluster numbers are group labels created by KMeans. Customers in the same cluster have similar behavior.")

    # This block handles app errors
    except Exception as exc:
        # This line shows the error message
        st.error(f"Prediction failed: {exc}")