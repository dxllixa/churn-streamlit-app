import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Churn Risk Detector", layout="centered")
st.title("📉 Customer Churn Risk Detector")
st.caption("Predict churn probability and risk level for a customer.")

@st.cache_resource
def train_model():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, list(X.columns)

rf, model_columns = train_model()

def risk_level(p: float) -> str:
    if p >= 0.7:
        return "High Risk 🔴"
    elif p >= 0.4:
        return "Medium Risk 🟡"
    else:
        return "Low Risk 🟢"

def prepare_input(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

st.subheader("Customer Details")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("SeniorCitizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=5)

with col2:
    phone = st.selectbox("PhoneService", ["Yes", "No"])
    multiple = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])

col3, col4 = st.columns(2)
with col3:
    online_sec = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
    device_prot = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])

with col4:
    stream_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
    stream_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
    payment = st.selectbox(
        "PaymentMethod",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=300.0, value=90.0)
    total = st.number_input("TotalCharges", min_value=0.0, max_value=20000.0, value=450.0)

if st.button("Predict Risk"):
    new_customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    X_new = prepare_input(new_customer)
    prob = float(rf.predict_proba(X_new)[:, 1][0])
    st.success(f"Churn Probability: **{prob:.2%}**")
    st.info(f"Risk Level: **{risk_level(prob)}**")
