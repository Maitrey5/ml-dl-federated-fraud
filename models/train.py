import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def train_model(
    model,
    X,
    y,
    scaler,
    epochs=2,
    batch_size=64,
    learning_rate=0.01,
    algorithm="fedavg",
    proximal_mu=0.0,
    global_parameters=None
):
    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for X_batch, y_batch in dataloader:

            logits = model(X_batch)

            # Normal classification loss
            loss = criterion(
                logits,
                y_batch
            )

            # ------------------------------------------
            # FedProx proximal term
            # ------------------------------------------
            if (
                algorithm == "fedprox"
                and global_parameters is not None
            ):

                proximal_term = 0.0

                for local_parameter, global_parameter in zip(
                    model.parameters(),
                    global_parameters
                ):
                    proximal_term += torch.sum(
                        (local_parameter - global_parameter) ** 2
                    )

                loss = loss + (
                    proximal_mu / 2.0
                ) * proximal_term

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(dataloader)
        )

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {average_loss:.4f}"
        )