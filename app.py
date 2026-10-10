import os
import joblib
import pandas as pd
import streamlit as st

# Configure the Streamlit page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction App")
st.markdown(
    """
    **CodSoft Machine Learning Internship - Task 3**  
    Predict whether a bank customer is likely to leave (**Churn**) or remain (**Retained**) 
    based on their demographic and financial profile using a trained **Logistic Regression Pipeline**.
    """
)

MODEL_FILE = "churn_model.pkl"

# Verify if model file exists
if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Model file `{MODEL_FILE}` not found in the current directory.")
    st.info(
        """
        **How to fix this:**
        1. Open and run `customer_churn_prediction.ipynb` in Google Colab (or locally).
        2. Download the generated `churn_model.pkl` file.
        3. Place `churn_model.pkl` in the same directory as this `app.py` file.
        4. Refresh this webpage.
        """
    )
    st.stop()

# Load the trained scikit-learn pipeline
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

model = load_model()

st.subheader("Customer Details")

# Input form for customer data
with st.form("churn_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("Country (Geography)", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=38, step=1)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
        tenure = st.slider("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)

    with col2:
        balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0, step=500.0)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0)
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"], index=0)
        estimated_salary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, max_value=300000.0, value=75000.0, step=500.0)

    # Convert binary dropdowns to numeric flags expected by the model
    has_cr_card_val = 1 if has_cr_card == "Yes" else 0
    is_active_member_val = 1 if is_active_member == "Yes" else 0

    submit_button = st.form_submit_button("Predict Churn")

if submit_button:
    # Build DataFrame matching the exact feature names and types from training
    input_data = pd.DataFrame([{
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_of_products),
        "HasCrCard": int(has_cr_card_val),
        "IsActiveMember": int(is_active_member_val),
        "EstimatedSalary": float(estimated_salary)
    }])

    # Predict outcome and probability
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    churn_probability = probabilities[1] * 100
    retention_probability = probabilities[0] * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Churn:** This customer is likely to leave the bank.")
    else:
        st.success(f"✅ **Low Risk:** This customer is likely to stay with the bank.")

    st.metric(label="Estimated Churn Probability", value=f"{churn_probability:.2f}%")
    st.progress(float(churn_probability / 100))

    with st.expander("🔍 View Input Summary"):
        st.dataframe(input_data, use_container_width=True)
