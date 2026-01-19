import os
import numpy as np
import h5py
import tensorflow as tf
from tqdm import tqdm
from model1 import create_fl_vibration_cnn  # simple CNN


# ---------------------------------------------------
# LOAD M01 TRAINING DATA  (same as FL client)
# ---------------------------------------------------
def load_m01_train():

    root = os.path.join("new_train", "M01")
    good_path = os.path.join(root, "good")
    bad_path  = os.path.join(root, "bad")

    X, y = [], []

    good_files = os.listdir(good_path) if os.path.exists(good_path) else []
    bad_files  = os.listdir(bad_path) if os.path.exists(bad_path) else []

    pbar = tqdm(total=len(good_files) + len(bad_files), desc="Loading M01 Train")

    # GOOD samples
    for f in good_files:
        with h5py.File(os.path.join(good_path, f), "r") as hf:
            X.append(hf["vibration_data"][:])
            y.append(1)
        pbar.update(1)

    # BAD samples
    for f in bad_files:
        with h5py.File(os.path.join(bad_path, f), "r") as hf:
            X.append(hf["vibration_data"][:])
            y.append(0)
        pbar.update(1)

    pbar.close()
    return np.array(X, np.float32), np.array(y, np.int32)


# Generic loader for a specific machine ID (ex: "M01") used by Prefect flows
def load_train_for(machine_id: str):
    root = os.path.join("new_train", machine_id)
    good_path = os.path.join(root, "good")
    bad_path = os.path.join(root, "bad")

    X, y = [], []

    good_files = os.listdir(good_path) if os.path.exists(good_path) else []
    bad_files = os.listdir(bad_path) if os.path.exists(bad_path) else []

    pbar = tqdm(total=len(good_files) + len(bad_files), desc=f"Loading {machine_id} Train")

    for f in good_files:
        with h5py.File(os.path.join(good_path, f), "r") as hf:
            X.append(hf["vibration_data"][:])
            y.append(1)
        pbar.update(1)

    for f in bad_files:
        with h5py.File(os.path.join(bad_path, f), "r") as hf:
            X.append(hf["vibration_data"][:])
            y.append(0)
        pbar.update(1)

    pbar.close()
    return np.array(X, np.float32), np.array(y, np.int32)



# ---------------------------------------------------
# LOAD GLOBAL TEST DATA (same as FL validation)
# ---------------------------------------------------
def load_global_test():

    root = "new_test"
    good_path = os.path.join(root, "good")
    bad_path  = os.path.join(root, "bad")

    X, y = [], []

    good_files = os.listdir(good_path) if os.path.exists(good_path) else []
    bad_files  = os.listdir(bad_path) if os.path.exists(bad_path) else []

    pbar = tqdm(total=len(good_files) + len(bad_files), desc="Loading Global Test")

    # GOOD samples
    for f in good_files:
        with h5py.File(os.path.join(good_path, f), "r") as hf:
            X.append(hf["vibration_data"][:])
            y.append(1)
        pbar.update(1)

    # BAD samples
    for f in bad_files:
        with h5py.File(os.path.join(bad_path, f), "r") as hf:
            X.append(hf["vibration_data"][:])
            y.append(0)
        pbar.update(1)

    pbar.close()
    return np.array(X, np.float32), np.array(y, np.int32)



# ---------------------------------------------------
# MAIN TRAINING LOOP — Matches FL (3 rounds = 3 epochs)
# ---------------------------------------------------
if __name__ == "__main__":
    
    print("\n📥 Loading M01-only training data…")
    X_train, y_train = load_m01_train()
    print("M01 Train samples:", len(X_train))

    print("\n📥 Loading Global Test data…")
    X_test, y_test = load_global_test()
    print("Global Test samples:", len(X_test))

    # Build simple CNN model
    model = create_fl_vibration_cnn ()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("\n🧠 Training ONLY on M01 (3 epochs to match FL rounds)…")
    model.fit(
        X_train, y_train,
        epochs=9,
        batch_size=32,
        verbose=1
    )

    print("\n🧪 Evaluating on GLOBAL TEST (same as FL eval)…")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print("\n=========================================")
    print(f"🎯 M01-ONLY GLOBAL TEST ACCURACY = {acc:.4f}")
    print("=========================================")
