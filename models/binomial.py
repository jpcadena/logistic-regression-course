"""
Binomial Logistic Regression.
"""
import numpy as np
from numpy import uint16, float16
from sklearn.linear_model import LogisticRegression


# Modelling
def train_model(
        x_train: np.ndarray, y_train: np.ndarray,
        inv_reg_strength: float16 = 1.0, solver: str = 'lbfgs',
        penalty: str = 'l2', max_iter: uint16 = 10000
) -> LogisticRegression:
    logistic_regression: LogisticRegression = LogisticRegression(
        penalty=penalty, C=inv_reg_strength, solver=solver, max_iter=max_iter)
    logistic_regression.fit(x_train, y_train)
    return logistic_regression
