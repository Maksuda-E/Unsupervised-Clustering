import json

import streamlit as st

from src.config import APP_TITLE
from src.predict import predict_admission

st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)
st.caption("Notebook-aligned MLP classifier for UCLA graduate admission prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gre_score = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
        university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
        lor = st.slider("LOR Strength", min_value=1.0, max_value=5.0, value=4.0, step=0.5)
        research = st.selectbox(
            "Research Experience",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    with col2:
        toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)
        sop = st.slider("SOP Strength", min_value=1.0, max_value=5.0, value=4.0, step=0.5)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.01)

    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = {
        "GRE_Score": gre_score,
        "TOEFL_Score": toefl_score,
        "University_Rating": university_rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research,
    }

    try:
        result = predict_admission(user_input)

        st.subheader(result["label"])

        if result["probability"] is not None:
            percent = result["probability"] * 100
            st.metric("Predicted probability for class = 1", f"{percent:.2f}%")
            st.progress(max(0, min(100, int(round(percent)))))
            st.caption(
                "This is the model's probability for the positive class produced by predict_proba."
            )
        else:
            st.info("Probability output is not available for this model.")

        with st.expander("Prediction details"):
            st.code(json.dumps(result, indent=2), language="json")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")