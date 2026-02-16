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

    # Custom strategy to save weights
    class CNCStrategy(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.latest_weights = None

        def aggregate_fit(self, server_round, results, failures):
            weights, metrics = super().aggregate_fit(server_round, results, failures)
            if weights is not None:
                self.latest_weights = fl.common.parameters_to_ndarrays(weights)
                print(f"Round {server_round} aggregation successful")
            return weights, metrics

        def save_model_weights(self, filename="final_model_weights.h5"):
            if self.latest_weights is not None:
                temp_model = create_fl_vibration_cnn()
                # Build model
                temp_model.predict(np.zeros((1, 1024, 3)), verbose=0)
                temp_model.set_weights(self.latest_weights)
                temp_model.save_weights(filename)
                print(f"Server weights saved to {filename}")

    # FedAvg Strategy
    strategy = CNCStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )

    # Save final weights
    print("\n[SERVER] Run finished!")
    strategy.save_model_weights("final_model_weights.weights.h5")

