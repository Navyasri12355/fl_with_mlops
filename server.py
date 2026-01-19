import os
import flwr as fl
import numpy as np
from model1 import create_fl_vibration_cnn   # simple CNN model

# Optional MLflow (guarded import)
try:
    import mlflow
    import mlflow.keras
except Exception:  # pragma: no cover
    mlflow = None


# ------------------------------------------------------------
# INITIAL GLOBAL MODEL WEIGHTS
# ------------------------------------------------------------
def get_initial_parameters():
    model = create_fl_vibration_cnn()

    # Build model by running one dummy prediction
    import numpy as _np
    vib_dummy = _np.zeros((1, 1024, 3), dtype=_np.float32)
    model.predict(vib_dummy, verbose=0)

    return model.get_weights()


# ------------------------------------------------------------
# START FLOWER SERVER
# ------------------------------------------------------------
if __name__ == "__main__":
    
    # Configuration from environment variables
    num_rounds = int(os.environ.get("FL_NUM_ROUNDS", "3"))
    min_clients = int(os.environ.get("FL_MIN_CLIENTS", "3"))
    server_address = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT", "federated")
    
    print(f"Starting Flower server (FedAvg)...")
    print(f"   Server address: {server_address}")
    print(f"   Number of rounds: {num_rounds}")
    print(f"   Minimum clients: {min_clients}")
    print(f"   MLflow experiment: {experiment_name}")
    
    # Setup MLflow if available
    if mlflow is not None:
        try:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            mlflow.set_experiment(experiment_name)
            print(f"MLflow configured: experiment={experiment_name}, uri={mlflow.get_tracking_uri()}")
            
            # Log server configuration
            with mlflow.start_run(run_name="fl_server_config"):
                mlflow.log_param("num_rounds", num_rounds)
                mlflow.log_param("min_clients", min_clients)
                mlflow.log_param("server_address", server_address)
                mlflow.log_param("strategy", "FedAvg")
                print("Server configuration logged to MLflow")
        except Exception as e:
            print(f"MLflow setup failed: {e}")
            mlflow = None

    # Convert initial Keras weights → Flower Parameters
    initial_weights = get_initial_parameters()
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)

    # FedAvg Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,             # use ALL clients every round
        fraction_evaluate=1.0,        # ask ALL clients to evaluate
        min_fit_clients=min_clients,  # require min_clients for training round
        min_evaluate_clients=min_clients,  # require min_clients for validation
        min_available_clients=min_clients,  # require min_clients connected clients
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )
