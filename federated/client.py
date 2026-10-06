from models.parameters import get_parameters, set_parameters
from models.train import train_model


class FederatedClient:

    def __init__(
        self,
        model,
        X,
        y,
        scaler
    ):

        self.model = model
        self.X = X
        self.y = y
        self.scaler = scaler

    def set_parameters(self, parameters):

        set_parameters(
            self.model,
            parameters
        )

    def get_parameters(self):

        return get_parameters(
            self.model
        )

    def fit(
        self,
        global_parameters,
        epochs=2
    ):

        # Load global model parameters
        self.set_parameters(
            global_parameters
        )

        # Train using the common scaler
        train_model(
            self.model,
            self.X,
            self.y,
            scaler=self.scaler,
            epochs=epochs
        )

        # Get updated local parameters
        updated_parameters = (
            self.get_parameters()
        )

        # Number of local samples
        num_samples = len(self.X)

        return (
            updated_parameters,
            num_samples
        )