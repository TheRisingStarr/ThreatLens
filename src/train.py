# --- Imports ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

# --- Paths ---
data_path = r"D:\CodeBase\ML Models\CyberAttack Project\data\training.csv"
model_dir = r"D:\CodeBase\ML Models\CyberAttack Project\model"
os.makedirs(model_dir, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(data_path)

# --- Drop unnecessary columns ---
cols_to_drop = [
    "id", "attack_cat", "is_sm_ips_ports", "sloss", "dloss",
    "ct_ftp_cmd", "is_ftp_login", "response_body_len", "ct_flw_http_mthd"
]
df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# --- Label Encoding (simple categorical columns) ---
proto_le = LabelEncoder()
df['proto'] = proto_le.fit_transform(df['proto'])

service_le = LabelEncoder()
df['service'] = service_le.fit_transform(df['service'])

# --- One-Hot Encoding (for multi-category column) ---
state_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
state_encoded = state_ohe.fit_transform(df[['state']])
state_encoded_cols = state_ohe.get_feature_names_out(['state'])
state_encoded_df = pd.DataFrame(state_encoded, columns=state_encoded_cols)

# Replace 'state' column with encoded features
df = pd.concat([df.drop('state', axis=1), state_encoded_df], axis=1)

# --- Final cleanup ---
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# --- Split features and labels ---
X = df.drop(columns=['label'])
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Columns to scale ---
numeric_cols = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
    'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
    'trans_depth', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_src_ltm',
    'ct_srv_dst'
]

# --- Standardization ---
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

# --- Train Logistic Regression Model ---
model = LogisticRegression(max_iter=10000, solver='saga')
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
pre = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# --- Save Model and Preprocessors ---
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(proto_le, os.path.join(model_dir, "proto_le.pkl"))
joblib.dump(service_le, os.path.join(model_dir, "service_le.pkl"))
joblib.dump(state_ohe, os.path.join(model_dir, "state_ohe.pkl"))

# --- Save Metadata ---
meta = {
    "cols_to_drop": cols_to_drop,
    "numeric_cols": numeric_cols,
    "cat_cols": ["proto", "service", "state"],
    "final_feature_order": list(X_train.columns)
}
joblib.dump(meta, os.path.join(model_dir, "meta.pkl"))

print("\nAll artifacts saved successfully.")
