import torch

from flwr.serverapp.strategy import FedAvg, FedProx

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg

from flwr.app import ArrayRecord, MetricRecord

from config.loader import load_config

from flower_app.task import (
    create_model,
    get_validation_data,
    get_test_data
)

from models.parameters import (
    get_parameters,
    set_parameters
)


app = ServerApp()

config = load_config()


# ==================================================
# Evaluation helper
# ==================================================

def evaluate_dataset(
    model,
    X,
    y,
    threshold
):

    model.eval()

    with torch.no_grad():

        logits = model(X)

        probabilities = torch.sigmoid(
            logits
        ).reshape(-1)

        predictions = (
            probabilities >= threshold
        ).int()

    y_true = y.numpy()

    y_pred = predictions.numpy()

    y_prob = probabilities.numpy()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm
    }


# ==================================================
# Global evaluation
# ==================================================

def evaluate_global_model(
    server_round,
    arrays
):

    model = create_model()

    parameters = arrays.to_numpy_ndarrays()

    set_parameters(
        model,
        parameters
    )

    threshold = config["evaluation"]["threshold"]

    # ==================================================
    # VALIDATION
    # ==================================================

    X_val, y_val = get_validation_data()

    validation_results = evaluate_dataset(
        model,
        X_val,
        y_val,
        threshold
    )

    print()
    print("========================================")
    print(
        f"GLOBAL VALIDATION - ROUND {server_round}"
    )
    print("========================================")

    print(
        f"Precision : "
        f"{validation_results['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{validation_results['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{validation_results['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{validation_results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{validation_results['pr_auc']:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(
        validation_results["confusion_matrix"]
    )

    print("========================================")
    print()

    # ==================================================
    # FINAL TEST
    #
    # Only execute after the FINAL round.
    # ==================================================

    num_rounds = config["federated"]["num_rounds"]

    if server_round == num_rounds:

        X_test, y_test = get_test_data()

        test_results = evaluate_dataset(
            model,
            X_test,
            y_test,
            threshold
        )

        print()
        print("########################################")
        print("FINAL TEST RESULTS")
        print("########################################")

        print(
            f"Precision : "
            f"{test_results['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{test_results['recall']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{test_results['f1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{test_results['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC    : "
            f"{test_results['pr_auc']:.4f}"
        )

        print()
        print("Confusion Matrix:")
        print(
            test_results["confusion_matrix"]
        )

        print("########################################")
        print()

    # ==================================================
    # Return validation metrics to Flower
    # ==================================================

    return MetricRecord(
        {
            "precision": float(
                validation_results["precision"]
            ),
            "recall": float(
                validation_results["recall"]
            ),
            "f1": float(
                validation_results["f1"]
            ),
            "roc_auc": float(
                validation_results["roc_auc"]
            ),
            "pr_auc": float(
                validation_results["pr_auc"]
            )
        }
    )


# ==================================================
# Server
# ==================================================

@app.main()
def main(grid, context):

    model = create_model()

    initial_parameters = get_parameters(
        model
    )

    initial_arrays = ArrayRecord(
        initial_parameters
    )

    # --------------------------------------------------
    # Read configuration
    # --------------------------------------------------

    algorithm = config["federated"]["algorithm"]

    if algorithm == "fedavg":

        strategy = FedAvg(
            fraction_train=config["federated"]["fraction_train"],
            fraction_evaluate=0.0,
            min_train_nodes=config["federated"]["min_train_nodes"],
            min_available_nodes=config["federated"]["min_available_nodes"]
        )

    elif algorithm == "fedprox":

        strategy = FedProx(
            fraction_train=config["federated"]["fraction_train"],
            fraction_evaluate=0.0,
            min_train_nodes=config["federated"]["min_train_nodes"],
            min_available_nodes=config["federated"]["min_available_nodes"],
            proximal_mu=config["fedprox"]["mu"]
        )

    else:

        raise ValueError(
            f"Unknown federated algorithm: {algorithm}"
        )

    # strategy = FedAvg(
    #     fraction_train=config["federated"]["fraction_train"],
    #     fraction_evaluate=0.0,
    #     min_train_nodes=config["federated"]["min_train_nodes"],
    #     min_available_nodes=config["federated"]["min_available_nodes"]
    # )

    # --------------------------------------------------
    # Start training
    # --------------------------------------------------

    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=config["federated"]["num_rounds"],
        evaluate_fn=evaluate_global_model
    )

    print()
    print("========================================")
    print("FEDERATED TRAINING COMPLETED")
    print("========================================")