import pandas as pd
from sklearn.model_selection import train_test_split

from config.loader import load_config


DATA_PATH = "creditcard.csv"


def load_data():

    config = load_config()

    seed = config["experiment"]["seed"]

    test_size = config["data"]["test_size"]

    validation_size = config["data"]["validation_size"]

    # --------------------------------------------------
    # Load complete dataset
    # --------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # --------------------------------------------------
    # FIRST split
    #
    # This creates the SAME test set as your
    # previous centralized experiment.
    # --------------------------------------------------

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )

    # --------------------------------------------------
    # SECOND split
    #
    # Split only the original training data.
    # Test data is never touched.
    # --------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        stratify=y_train_full,
        random_state=seed
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )