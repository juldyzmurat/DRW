#%%
import gc
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten, Input, InputLayer
from tensorflow.keras.models import Model, Sequential, load_model
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa

#%%
# Load Data
df_train = pd.read_parquet("train.parquet")
df_test = pd.read_parquet("test.parquet")

train = df_train
test = df_test

np.random.seed(0)

#drop the timestep columns, which is the index column 
train = train.reset_index(drop=True)

#drop columns that have inf values 
inf_per_column = np.isinf(train).sum()
inf_per_column = inf_per_column[inf_per_column>0]
train = train.drop(columns=inf_per_column.index.tolist())
#list of inf columns was also saved as inf_columns.txt to be used in later sessions 

#drop the label column from the features variable 
X = train.drop(columns=['label'])
#label assigned to the label variable
y = train['label']

#split the data for training and testing the autoencoder 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#normalize the data to make it suitable for the autoencoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

#define the input dimension for the autoencoder
#this is the number of features in the training data
input_dim = X_scaled.shape[1]
print("Input dim:",input_dim)
#we want to encode the data into a lower dimensional space
#in this case, we will use 30 dimensions because the original data has over 800 features 
#further reducing it to 10 dimensions may have caused too much information loss
latent_dim = 30

#define the autoencoder model
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(latent_dim, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)

#we compile the model with the Adam optimizer and use mean squared error to measure loss
autoencoder.compile(optimizer='adam', loss='mse')

#we also use early stopping to prevent overfitting and make training faster
early_stop = EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

# Train the autoencoder
print("[INFO] Training autoencoder...")
start = time.time()
#we could have used more epochs, but 50 is a max my local machine can handle without running out of memory 
#use of parallel processing is not recommended here as it may cause memory issues if your RAM is limited
#the batch size is set to 32, which is a common choice for training neural networks, given the large size of the dataset
#even smaller batch sizes can lead to overfitting, so we keep it at 32
autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

#the trained autoencoder model is saved to a file for later use
#because re-training the model every time is not efficient
autoencoder.save("autoencoder_model-2.h5")

#%%
#now we need to turn our original data into a lower dimensional space
#we will use the encoder part of the autoencoder model to do this
#however, it is a big load for the memory to load the entire dataset at once and also to encode it
#so we will split the data into smaller chunks and encode them one by one
#this is done to avoid memory issues and make the process more efficient
PARQUET_PATH = "train.parquet"
SAVE_DIR = "train_chunks"
BATCH_SIZE = 10000

# Create output folder if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Open the Parquet file
parquet_file = pq.ParquetFile(PARQUET_PATH)
batch_id = 0

print("⚙️ Splitting into chunks of ~10000 rows...")

# Loop through each row group (efficient, works well for large files)
for row_group_index in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(row_group_index)
    num_rows = table.num_rows

    for start in range(0, num_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_rows)
        chunk = table.slice(start, end - start)

        output_path = os.path.join(SAVE_DIR, f"chunk_{batch_id:03d}.parquet")
        pq.write_table(chunk, output_path)
        print(f"🧩 Wrote: {output_path} [{start}:{end}]")

        batch_id += 1

print(f"✅ Done. Wrote {batch_id} chunks to '{SAVE_DIR}'")

#%%
# Now we will encode each chunk using the trained autoencoder model
# and save the encoded data
CHUNKS_DIR = "train_chunks"
autoencoder_full = load_model("autoencoder_model-2.h5", compile=False)
encoder = Model(inputs=autoencoder_full.input, outputs=autoencoder_full.layers[2].output)

# === Setup ===
columns_to_drop = [
    'X697', 'X698', 'X699', 'X700', 'X701', 'X702', 'X703',
    'X704', 'X705', 'X706', 'X707', 'X708', 'X709', 'X710',
    'X711', 'X712', 'X713', 'X714', 'X715', 'X716', 'X717'
]
scaler = StandardScaler()
fitted_scaler = False

X_latent_all = []
y_all = []

# === Process each chunk ===
chunk_files = sorted(f for f in os.listdir(CHUNKS_DIR) if f.endswith(".parquet"))

for chunk_file in tqdm(chunk_files, desc="🔄 Processing chunks", unit="chunk"):
    path = os.path.join(CHUNKS_DIR, chunk_file)
    df = pd.read_parquet(path)

    # Drop inf columns
    inf_cols = df.columns[np.isinf(df).sum() > 0]
    df = df.drop(columns=inf_cols)

    # Drop unwanted columns
    df = df.drop(columns=columns_to_drop, errors='ignore')

    if 'label' not in df.columns:
        continue

    df = df.dropna(subset=['label'])
    y_chunk = df['label'].values
    X_chunk = df.drop(columns=['label'], errors='ignore').values

    if not fitted_scaler:
        scaler.fit(X_chunk)
        fitted_scaler = True

    X_scaled = scaler.transform(X_chunk)
    X_latent = encoder.predict(X_scaled, verbose=0)

    X_latent_all.append(X_latent)
    y_all.append(y_chunk)

    # Free memory
    del df, X_chunk, y_chunk, X_scaled, X_latent
    gc.collect()

# === Final merge ===
X_all = np.vstack(X_latent_all)
y_all = np.concatenate(y_all)

print("🔀 Performing final train/test split...")
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# === Save final arrays (can change to .parquet if needed) ===
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
y_train_df = pd.DataFrame(y_train, columns=["label"])
y_test_df = pd.DataFrame(y_test, columns=["label"])

# === Save to CSV ===
X_train_df.to_csv("X_train.csv", index=False)
X_test_df.to_csv("X_test.csv", index=False)
y_train_df.to_csv("y_train.csv", index=False)
y_test_df.to_csv("y_test.csv", index=False)

print("✅ All done.")
print(f"🧠 X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"🎯 y_train: {y_train.shape}, y_test: {y_test.shape}")

#%%
#no we will use the encoded data to train our model to predict the labels
#we will use a simple CNN for this task
#we will also use early stopping to prevent overfitting and make training faster
# Load features and labels
X_train = pd.read_csv("X_train.csv").values
X_test = pd.read_csv("X_test.csv").values
y_train = pd.read_csv("y_train.csv").values.ravel()  # ravel to flatten (1D)
y_test = pd.read_csv("y_test.csv").values.ravel()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Reshape input for Conv1D: (samples, timesteps, features)
# Here, treat each sample as 1 "time step" with latent_dim features
X_train_cnn = X_train[..., np.newaxis]  # shape: (n_samples, latent_dim, 1)
X_test_cnn = X_test[..., np.newaxis]

# Define CNN model
cnn_model = Sequential([
    InputLayer(input_shape=(X_train_cnn.shape[1], 1)),
    Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1)  # Output layer for regression
])

cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train with timing
print("[CNN] Training started...")
start = time.time()
history = cnn_model.fit(
    X_train_cnn, y_train,
    validation_data=(X_test_cnn, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
end = time.time()
print(f"[CNN] Training finished in {end - start:.2f} seconds.")

cnn_model.save("cnn_model.keras")  

#test the model on the test set (of the trianing data that has labels)
print("\nEvaluation on test set:")
y_pred = cnn_model.predict(X_test_cnn).ravel()
mse = mean_squared_error(y_test, y_pred)
print(f"MSE = {mse:.4f}")
#%%
#now we can use the trained CNN model to predict the labels for the test set
#however, due to the large size of the test set, we will split it into smaller chunks and predict them one by one
SAVE_DIR = "split_chunks_arrow"

# define the batch size for splitting the data, 6000 corresponds to 1% of the original dataset size
#this is a good size to balance memory usage and processing time
BATCH_SIZE = 6000
os.makedirs(SAVE_DIR, exist_ok=True)

# Open the original Parquet file
parquet_file = pq.ParquetFile("test.parquet")

# Initialize batch ID counter
batch_id = 0

print("⚙️ Splitting into chunks of ~6000 rows...")

# Memory efficient: calculate IDs on-the-fly for each chunk
# Start from 1 as specified, this is important to prevent the loss of original IDs
current_global_id = 1  

# Loop through each row group (memory efficient - one row group at a time)
for row_group_index in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(row_group_index)
    num_rows = table.num_rows
    
    # Chunk this row group into BATCH_SIZE-row tables
    for start in range(0, num_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_rows)
        chunk = table.slice(start, end - start)
        chunk_size = chunk.num_rows
        
        # Calculate the exact ID range for this chunk
        chunk_start_id = current_global_id + start
        chunk_end_id = chunk_start_id + chunk_size - 1
        
        # Create ID array only for this chunk (memory efficient)
        chunk_ids = pa.array(range(chunk_start_id, chunk_start_id + chunk_size))
        
        # Add ID as first column
        chunk_with_id = chunk.add_column(0, "id", chunk_ids)
        
        print(f"Chunk {batch_id:03d}: {chunk_size} rows, ID range: {chunk_start_id} to {chunk_end_id}")
        
        # Write immediately to free memory
        pq.write_table(chunk_with_id, f"{SAVE_DIR}/chunk_{batch_id:03d}.parquet")
        batch_id += 1
        
        # Clean up references to help garbage collection
        #this is important to prevent memory leaks 
        del chunk_ids, chunk_with_id
    
    # Update global ID counter for next row group
    current_global_id += num_rows
    
    # Clean up row group table
    del table

print(f"✅ Done. Wrote {batch_id} chunks to {SAVE_DIR}")
# %%
#after this step we run the process_chunk.py script for each chunk
#for this we will use the command line to run the script
#since we have many chunks, we can use a simple bash script to run the process_chunk.py script for each chunk
#the bash script is names run_all_chunks.sh
#we just need to run it in the terminal like this:
# bash run_all_chunks.sh
# %%
#after running the process_chunk.py script for each chunk, we will have many CSV files in the predicted_chunks folder
#we will combine them into a single CSV file

# Get all CSV files in the folder, sorted
csv_files = sorted(glob.glob("predicted_chunks/chunk_*.csv"))

# Load and concatenate all CSVs
df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Save the final combined CSV
df_all.to_csv("sub_predictions.csv", index=False)

print("✅ Combined all chunk CSVs into sub_predictions.csv")