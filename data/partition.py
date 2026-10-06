
import numpy as np


def create_clients(
    X_train,
    y_train,
    num_clients=10,
    seed=42
):

    rng = np.random.default_rng(seed)

    indices = np.arange(
        len(X_train)
    )

    rng.shuffle(indices)

    client_indices = np.array_split(
        indices,
        num_clients
    )

    clients = {}

    for client_id, indices in enumerate(
        client_indices
    ):

        clients[client_id] = {
            "X": X_train.iloc[indices].copy(),
            "y": y_train.iloc[indices].copy()
        }

    return clients