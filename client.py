import os
import sys
import numpy as np
import h5py
import flwr as fl
import tensorflow as tf
from tqdm import tqdm

from model1 import create_fl_vibration_cnn  # simple CNN only
from concurrent.futures import ThreadPoolExecutor

# Reduce TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Disable tqdm if running in pipeline mode
IS_PIPELINE = os.environ.get("FL_PIPELINE_MODE", "False") == "True"
if IS_PIPELINE:
    def silent_tqdm(iterable, *args, **kwargs):
        return iterable
else:
    silent_tqdm = tqdm

# Optional MLflow (guarded import)
try:
    import mlflow
    import mlflow.keras
except Exception:  # pragma: no cover
    mlflow = None

# Use a default experiment name which can be overridden by MLFLOW_EXPERIMENT env var
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "federated")
if mlflow is not None:
    try:
        # Set tracking URI if provided
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        print(f"MLflow configured: experiment={MLFLOW_EXPERIMENT}, uri={mlflow.get_tracking_uri()}")
    except Exception as e:
        print(f"MLflow setup failed: {e}")
        mlflow = None



# ============================================================
# LOAD TRAINING DATA (machine-specific)
# ============================================================
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# LOAD TRAINING DATA (machine-specific)
# ============================================================
def load_file(path, label):
    with h5py.File(path, "r") as f:
        return f["vibration_data"][:], label

def load_train_data(machine_id):
    # Check for consolidated file first
    consolidated_path = os.path.join("data_consolidated", f"{machine_id}_consolidated.h5")
    if os.path.exists(consolidated_path):
        print(f"Loading consolidated training data for {machine_id} from {consolidated_path}...")
        with h5py.File(consolidated_path, 'r') as f:
            X = f['vibration_data'][:]
            y = f['label'][:]
            return X.astype(np.float32), y.astype(np.int32)

    root = os.path.join("new_train", machine_id)

    if not os.path.exists(root):
        raise FileNotFoundError(f"ERROR: Folder not found: {root}")

    good_path = os.path.join(root, "good")
    bad_path  = os.path.join(root, "bad")

    good_files = [os.path.join(good_path, f) for f in os.listdir(good_path) if f.endswith(".h5")]
    bad_files  = [os.path.join(bad_path, f) for f in os.listdir(bad_path) if f.endswith(".h5")]
    
    all_files = [(f, 1) for f in good_files] + [(f, 0) for f in bad_files]
    
    print(f"Loading {len(all_files)} training files for {machine_id} using threads...")
    
    X, y = [], []
    
    # Use threads to speed up I/O
    with ThreadPoolExecutor(max_workers=8) as executor:
        # submit all tasks
        futures = [executor.submit(load_file, f, l) for f, l in all_files]
        
        # process as they complete
        for future in silent_tqdm(futures, desc=f"Train {machine_id}", total=len(all_files)):
            data, label = future.result()
            X.append(data)
            y.append(label)

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)




# ============================================================
# LOAD GLOBAL VALIDATION DATA (shared across all clients)
# ============================================================
def load_global_val():
    # Check for consolidated file first
    consolidated_path = os.path.join("data_consolidated", "test_consolidated.h5")
    if os.path.exists(consolidated_path):
        print(f"Loading consolidated global validation data from {consolidated_path}...")
        with h5py.File(consolidated_path, 'r') as f:
            X = f['vibration_data'][:]
            y = f['label'][:]
            return X.astype(np.float32), y.astype(np.int32)

    root = "new_test"

    good_path = os.path.join(root, "good")
    bad_path  = os.path.join(root, "bad")

    good_files = [os.path.join(good_path, f) for f in os.listdir(good_path) if f.endswith(".h5")]
    bad_files  = [os.path.join(bad_path, f) for f in os.listdir(bad_path) if f.endswith(".h5")]
    
    all_files = [(f, 1) for f in good_files] + [(f, 0) for f in bad_files]

    print(f"Loading {len(all_files)} validation files using threads...")

    X, y = [], []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_file, f, l) for f, l in all_files]
        
        for future in silent_tqdm(futures, desc="Global Validation", total=len(all_files)):
            data, label = future.result()
            X.append(data)
            y.append(label)

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)




# ============================================================
# FEDERATED LEARNING CLIENT
# ============================================================
class CNCClient(fl.client.NumPyClient):

    def __init__(self, machine_num):

        self.machine_id = f"M0{machine_num}"
        print(f"\nStarting Client for {self.machine_id}")

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

        # MLflow per-client round counting
        self.round = 0
        self.eval_round = 0

        # If mlflow is available we will create a run for each round in fit/evaluate
        self.mlflow = mlflow
        if self.mlflow is not None:
            try:
                self.mlflow.set_experiment(MLFLOW_EXPERIMENT)
            except Exception:
                pass
        
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)


    # Flower API
    def get_parameters(self, config=None):
        return self.model.get_weights()

    def set_parameters(self, params):
        self.model.set_weights(params)

    def fit(self, parameters, config):
        """Run a single training round. We log both the global model (before local training)
        and the local updated model (after training) to MLflow per round along with artifacts.
        """
        self.round += 1
        round_id = self.round

        print(f"\n{self.machine_id}: Received global weights (round {round_id})")
        self.set_parameters(parameters)

        # Evaluate global model on local validation set (pre-training)
        g_loss, g_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)

        # Start an MLflow run for this client round
        if self.mlflow is not None:
            try:
                self.mlflow.start_run(run_name=f"{self.machine_id}-round-{round_id}")
                self.mlflow.log_param("round", round_id)
                self.mlflow.log_param("client", self.machine_id)
                self.mlflow.log_metric("global_loss", float(g_loss), step=round_id)
                self.mlflow.log_metric("global_accuracy", float(g_acc), step=round_id)

                # Log produced artifact: save a small file with pre-train metrics
                pre_metrics_path = os.path.join("reports", f"client_{self.machine_id}_round_{round_id}_pre_metrics.txt")
                with open(pre_metrics_path, "w") as fh:
                    fh.write(f"global_loss={g_loss}\nglobal_accuracy={g_acc}\n")
                self.mlflow.log_artifact(pre_metrics_path)
            except Exception as exc:
                print("WARNING: MLflow logging (pre-train) failed:", exc)

        # Train locally (several epochs for meaningful update)
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )

        # Evaluate local model on same validation set (post-training)
        l_loss, l_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)

        # Log post-train metrics and model artifact
        if self.mlflow is not None:
            try:
                self.mlflow.log_metric("local_loss", float(l_loss), step=round_id)
                self.mlflow.log_metric("local_accuracy", float(l_acc), step=round_id)

                # Which performed better on the local validation set?
                local_better = 1 if l_acc > g_acc else 0
                self.mlflow.log_metric("local_better", local_better, step=round_id)

                # Log the Keras model weights as an artifact (more reliable than log_model)
                try:
                    weights_filename = f"local_model_{self.machine_id}_round_{round_id}.weights.h5"
                    self.model.save_weights(weights_filename)
                    self.mlflow.log_artifact(weights_filename)
                    os.remove(weights_filename)
                    print(f"      {self.machine_id}: Logged model weights to MLflow.")
                except Exception as exc:
                    print(f"      WARNING: {self.machine_id}: Failed to log weights: {exc}")

                # Save a simple report artifact
                report_path = os.path.join("reports", f"client_{self.machine_id}_round_{round_id}_report.txt")
                with open(report_path, "w") as fh:
                    fh.write(f"global_accuracy={g_acc}\nlocal_accuracy={l_acc}\nlocal_better={local_better}\n")
                self.mlflow.log_artifact(report_path)

            except Exception as exc:
                print("WARNING: MLflow logging (post-train) failed:", exc)
            finally:
                try:
                    self.mlflow.end_run()
                except Exception:
                    pass

        return self.get_parameters(), len(self.y_train), {"local_accuracy": float(l_acc)}

    def evaluate(self, parameters, config):
        # Called by the server to evaluate the GLOBAL model on this client's validation set.
        self.eval_round += 1
        eval_id = self.eval_round

        self.set_parameters(parameters)

        loss, acc = self.model.evaluate(
            self.X_val,
            self.y_val,
            verbose=0
        )

        print(f"{self.machine_id} | Eval accuracy = {acc:.4f}")

        # Log evaluation as an MLflow run too
        if self.mlflow is not None:
            try:
                self.mlflow.start_run(run_name=f"{self.machine_id}-eval-round-{eval_id}")
                self.mlflow.log_param("client", self.machine_id)
                self.mlflow.log_param("eval_round", eval_id)
                self.mlflow.log_metric("global_eval_loss", float(loss), step=eval_id)
                self.mlflow.log_metric("global_eval_accuracy", float(acc), step=eval_id)

                # Save small artifact
                path = os.path.join("reports", f"client_{self.machine_id}_eval_{eval_id}.txt")
                with open(path, "w") as fh:
                    fh.write(f"global_eval_loss={loss}\nglobal_eval_accuracy={acc}\n")
                self.mlflow.log_artifact(path)
            except Exception as exc:
                print("WARNING: MLflow eval logging failed:", exc)
            finally:
                try:
                    self.mlflow.end_run()
                except Exception:
                    pass

        return float(loss), len(self.X_val), {"accuracy": float(acc)}



# ============================================================
# RUN CLIENT
# ============================================================
if __name__ == "__main__":
    client_number = int(sys.argv[1])
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=CNCClient(client_number).to_client()
    )
