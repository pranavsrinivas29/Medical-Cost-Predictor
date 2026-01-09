import os
import requests
import streamlit as st

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("Insurance Charges Predictor")
st.caption("Enter details and get a predicted insurance charge.")

with st.form("predict_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

    sex = st.selectbox("Sex", options=["female", "male"])
    bmi = st.number_input("BMI", min_value=1.0, max_value=80.0, value=27.5, step=0.1, format="%.1f")
    children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)

    smoker = st.selectbox("Smoker", options=["no", "yes"])
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region,
    }

    try:
        with st.spinner("Calling model..."):
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            pred = data.get("prediction", None)
            if pred is None:
                st.error("Response did not contain 'prediction'.")
            else:
                st.success(f"Predicted Charges: {pred:,.2f}")
        else:
            # FastAPI returns details in {"detail": "..."}
            try:
                err = resp.json()
            except Exception:
                err = {"detail": resp.text}
            st.error(f"API Error ({resp.status_code}): {err.get('detail', err)}")

    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to API at {API_URL}. "
            "Make sure FastAPI is running: `uvicorn app.main:app --reload`"
        )
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
