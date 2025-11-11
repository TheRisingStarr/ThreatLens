import streamlit as st
import pandas as pd
import joblib

# --- Load artifacts ---
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
proto_le = joblib.load("model/proto_le.pkl")
service_le = joblib.load("model/service_le.pkl")
state_ohe = joblib.load("model/state_ohe.pkl")
meta = joblib.load("model/meta.pkl")

st.title("Cyberattack Detection using Logistic Regression")
st.write("Upload your CSV file or input manually below:")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

def preprocess(df):
    df['proto'] = proto_le.transform(df['proto'])
    df['service'] = service_le.transform(df['service'])
    state_encoded = state_ohe.transform(df[['state']])
    state_cols = state_ohe.get_feature_names_out(['state'])
    df = pd.concat([df.drop('state', axis=1), pd.DataFrame(state_encoded, columns=state_cols)], axis=1)
    df = df.reindex(columns=meta["final_feature_order"], fill_value=0)
    df[meta["numeric_cols"]] = scaler.transform(df[meta["numeric_cols"]])
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview:", df.head())
    
    X = preprocess(df)
    preds = model.predict(X)
    st.write("### Predictions:")
    st.write(preds)
else:
    st.info("Upload a CSV file to get predictions.")
