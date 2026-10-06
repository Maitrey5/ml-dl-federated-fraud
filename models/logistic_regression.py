from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:

    def __init__(self, class_weight=None):

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic_regression", LogisticRegression(
                class_weight=class_weight,
                max_iter=10000,
                random_state=42
            ))
        ])

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]