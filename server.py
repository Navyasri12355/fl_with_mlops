import flwr as fl
import numpy as np
from model1 import create_fl_vibration_cnn   # simple CNN model


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

    # Convert initial Keras weights → Flower Parameters
    initial_weights = get_initial_parameters()
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)

    # FedAvg Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,             # use ALL clients every round
        fraction_evaluate=1.0,        # ask ALL clients to evaluate
        min_fit_clients=3,            # require 3 clients for training round
        min_evaluate_clients=3,       # require 3 clients for validation
        min_available_clients=3,      # require 3 connected clients
        initial_parameters=initial_parameters,
    )

    print("🚀 Starting Flower server (FedAvg)...")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )
