from models.parameters import get_parameters, set_parameters
from models.train import train_model


class FederatedClient:

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def get_parameters(self):
        return get_parameters(self.model)

    def fit(self, global_parameters, epochs=2):
        # Load global model parameters
        self.set_parameters(global_parameters)

        # Train on local data
        train_model(
            self.model,
            self.X,
            self.y,
            epochs=epochs
        )

        # Get updated parameters
        updated_parameters = self.get_parameters()

        # Number of local training samples
        num_samples = len(self.X)

        return updated_parameters, num_samples