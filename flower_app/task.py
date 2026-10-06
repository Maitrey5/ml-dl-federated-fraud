import torch
from sklearn.preprocessing import StandardScaler

from config.loader import load_config

from data.data_loader import load_data
from data.partition import create_clients

from models.logistic_regression_torch import LogisticRegressionTorch
from models.train import train_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = load_config()

NUM_CLIENTS = config["federated"]["num_clients"]


# --------------------------------------------------
# Load data
# --------------------------------------------------

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
) = load_data()


# --------------------------------------------------
# Global scaler
#
# Fit ONLY on training data.
# --------------------------------------------------

scaler = StandardScaler()

scaler.fit(X_train)


# --------------------------------------------------
# Partition training data among clients
# --------------------------------------------------

clients_data = create_clients(
    X_train,
    y_train,
    num_clients=NUM_CLIENTS,
    seed=config["experiment"]["seed"]
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

def create_model():

    return LogisticRegressionTorch(
        input_dim=X_train.shape[1]
    )


# --------------------------------------------------
# Get client data
# --------------------------------------------------

def get_client_data(client_id):

    client_data = clients_data[client_id]

    return (
        client_data["X"],
        client_data["y"]
    )


# --------------------------------------------------
# Train local model
# --------------------------------------------------
def train_client(
    model,
    X,
    y,
    algorithm="fedavg",
    proximal_mu=0.0,
    global_parameters=None
):
    train_model(
        model,
        X,
        y,
        scaler=scaler,
        epochs=config["model"]["local_epochs"],
        batch_size=config["model"]["batch_size"],
        learning_rate=config["model"]["learning_rate"],
        algorithm=algorithm,
        proximal_mu=proximal_mu,
        global_parameters=global_parameters
    )

    


# --------------------------------------------------
# Prepare validation data
# --------------------------------------------------

def get_validation_data():

    X_val_scaled = scaler.transform(X_val)

    X_val_tensor = torch.tensor(
        X_val_scaled,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val.values,
        dtype=torch.float32
    )

    return (
        X_val_tensor,
        y_val_tensor
    )


# --------------------------------------------------
# Prepare final test data
# --------------------------------------------------

def get_test_data():

    X_test_scaled = scaler.transform(X_test)

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test.values,
        dtype=torch.float32
    )

    return (
        X_test_tensor,
        y_test_tensor
    )