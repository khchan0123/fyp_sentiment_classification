import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

BASE_DIR = "modelling" 

# Load DL-ready datasets
print("Loading balanced datasets...")
train_df = pd.read_csv(f"{BASE_DIR}/data/train_dl.csv")
test_df = pd.read_csv(f"{BASE_DIR}/data/test_dl.csv")

X_train = train_df["text"].fillna("").astype(str).tolist()
y_train = train_df["label"].tolist()

X_test = test_df["text"].fillna("").astype(str).tolist()
y_test = test_df["label"].tolist()

print("Encoding labels...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Save the Encoder 
joblib.dump(le, f"{BASE_DIR}/features/label_encoder.pkl")

# Save encoded DL datasets
pd.DataFrame({"text": X_train, "label": y_train_enc}).to_csv(
    f"{BASE_DIR}/data/train_dl_encoded.csv", index=False
)

pd.DataFrame({"text": X_test, "label": y_test_enc}).to_csv(
    f"{BASE_DIR}/data/test_dl_encoded.csv", index=False
)

print(f"Deep learning datasets prepared.")
print(f"   -> Train Size: {len(X_train)}")
print(f"   -> Test Size:  {len(X_test)}")
print(f"   -> Classes:    {le.classes_}")