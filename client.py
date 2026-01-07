import os
import sys
import numpy as np
import h5py
import flwr as fl
import tensorflow as tf
from tqdm import tqdm

from model1 import create_fl_vibration_cnn  # simple CNN only


# ============================================================
# LOAD TRAINING DATA (machine-specific)
# ============================================================
def load_train_data(machine_id):

    root = os.path.join("new_train", machine_id)

    if not os.path.exists(root):
        raise FileNotFoundError(f"❌ Folder not found: {root}")

    good_path = os.path.join(root, "good")
    bad_path  = os.path.join(root, "bad")

    X, y = [], []

    good_files = [f for f in os.listdir(good_path) if f.endswith(".h5")]
    bad_files  = [f for f in os.listdir(bad_path) if f.endswith(".h5")]

    pbar = tqdm(total=len(good_files) + len(bad_files), desc=f"Train {machine_id}")

    for file in good_files:
        with h5py.File(os.path.join(good_path, file), "r") as f:
            X.append(f["vibration_data"][:])
            y.append(1)
        pbar.update(1)

    for file in bad_files:
        with h5py.File(os.path.join(bad_path, file), "r") as f:
            X.append(f["vibration_data"][:])
            y.append(0)
        pbar.update(1)

    pbar.close()

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)




# ============================================================
# LOAD GLOBAL VALIDATION DATA (shared across all clients)
# ============================================================
def load_global_val():

    root = "new_test"

    good_path = os.path.join(root, "good")
    bad_path  = os.path.join(root, "bad")

    X, y = [], []

    good_files = [f for f in os.listdir(good_path) if f.endswith(".h5")]
    bad_files  = [f for f in os.listdir(bad_path) if f.endswith(".h5")]

    pbar = tqdm(total=len(good_files) + len(bad_files), desc="Global Validation")

    for file in good_files:
        with h5py.File(os.path.join(good_path, file), "r") as f:
            X.append(f["vibration_data"][:])
            y.append(1)
        pbar.update(1)

    for file in bad_files:
        with h5py.File(os.path.join(bad_path, file), "r") as f:
            X.append(f["vibration_data"][:])
            y.append(0)
        pbar.update(1)

    pbar.close()

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)




# ============================================================
# FEDERATED LEARNING CLIENT
# ============================================================
class CNCClient(fl.client.NumPyClient):

    def __init__(self, machine_num):

        self.machine_id = f"M0{machine_num}"
        print(f"\n🔧 Starting Client for {self.machine_id}")

        # Load train + validation
        self.X_train, self.y_train = load_train_data(self.machine_id)
        self.X_val,   self.y_val   = load_global_val()

        # Build simple CNN
        self.model = create_fl_vibration_cnn ()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )


    # Flower API
    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, params):
        self.model.set_weights(params)

    def fit(self, parameters, config):
        print(f"\n📡 {self.machine_id}: Received global weights")
        self.set_parameters(parameters)

        # Train exactly 1 epoch
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=3,
            batch_size=32,
            verbose=1
        )

        return self.get_parameters(), len(self.y_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, acc = self.model.evaluate(
            self.X_val,
            self.y_val,
            verbose=0
        )

        print(f"📊 {self.machine_id} | Eval accuracy = {acc:.4f}")

        return float(loss), len(self.X_val), {"accuracy": float(acc)}



# ============================================================
# RUN CLIENT
# ============================================================
if __name__ == "__main__":
    client_number = int(sys.argv[1])
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=CNCClient(client_number)
    )
