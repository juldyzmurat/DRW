import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
import joblib  # for rf_model
import sys
import os
import numpy as np
from pathlib import Path


# === CONFIG ===
CHUNK_ID = int(sys.argv[1])  # e.g., pass 0, 1, 2...99 as command-line arg
CHUNK_PATH = f"split_chunks_arrow/chunk_{CHUNK_ID:03d}.parquet"
RF_MODEL_PATH = "cnn_model.joblib"
AE_MODEL_PATH = "autoencoder_model-2.h5"
OUTPUT_PATH = f"predicted_chunks/chunk_{CHUNK_ID:03d}.csv"

# === SETUP ===
os.makedirs("predicted_chunks", exist_ok=True)

# === LOAD ===
df = pd.read_parquet(CHUNK_PATH)
ae = load_model(AE_MODEL_PATH, compile=False)
# Extract the encoder part (latent output = layer 2)
ae = Model(inputs=ae.input, outputs=ae.layers[2].output)

cnn_model = joblib.load(RF_MODEL_PATH)

# === Store 'id' and remove from df ===
id_column = df["id"].copy()


# === PREPROCESS ===
INF_FILE = "inf_columns.txt"

cols_from_file = []
p = Path(INF_FILE)
if p.exists():
    with p.open("r") as f:
        cols_from_file = [line.strip() for line in f if line.strip()]

columns_to_drop = list(dict.fromkeys(["label", *cols_from_file, "id"]))

df = df.drop(columns=columns_to_drop, errors='ignore')

# === SCALE ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# === PREDICT ===
latent = ae.predict(X_scaled, verbose=0)
preds = cnn_model.predict(latent).flatten()

# === OUTPUT ===
out_df = pd.DataFrame({
    "id": id_column.values,
    "prediction": preds
})
out_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Finished processing chunk {CHUNK_ID}")
print(f"✅ Processed chunk {CHUNK_ID}, saved to {OUTPUT_PATH}")
