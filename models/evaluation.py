"""
Main script
"""
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def get_model_metrics(
        logistic_regression: LogisticRegression, x_test: np.ndarray,
        y_test: np.ndarray) -> np.ndarray:
    print(logistic_regression.score(x_test, y_test))
    print(logistic_regression.predict_proba(x_test))
    print(logistic_regression.coef_)
    conf_matrix: np.ndarray = confusion_matrix(
        y_test, logistic_regression.predict(x_test))
    print(conf_matrix)
    return conf_matrix


def model_report(
        y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]
) -> str:
    report: str = classification_report(
        y_true, y_pred, target_names=target_names)
    print(report)
    return report


def roc_curve(
        logistic_regression: LogisticRegression, x_test: np.ndarray,
        y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y_score = logistic_regression.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    roc_auc_score = metrics.roc_auc_score(y_true, y_score)
    return fpr, tpr, roc_auc_score
