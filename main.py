"""
Main script for Logistic Regression Course
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from cleaning.preprocessing import fill_and_update, split_data
from cleaning.transformation import one_hot_encoding, scaling
from core.config import MAX_COLUMNS, WIDTH, FILENAME, CHUNK_SIZE, DTYPES, \
    converters, DEPENDENT_COLUMN, COLORS, TEST_SIZE, RANDOM_STATE, \
    INVERSE_REGULARIZATION_STRENGTH, SOLVER, PENALTY
from engineering.persistence_manager import PersistenceManager, DataPath
from engineering.preliminar_analysis import data_analyze
from engineering.visualization import plot_confusion_matrix, plot_weights, \
    plot_distribution, plot_count, plot_roc_curve
from models.binomial import train_model
from models.evaluation import get_model_metrics, model_report, roc_curve

print(MAX_COLUMNS, type(MAX_COLUMNS))
print(WIDTH, type(WIDTH))
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', WIDTH)


if __name__ == '__main__':
    raw: DataPath = DataPath.RAW
    processed: DataPath = DataPath.PROCESSED
    raw_dataframe = PersistenceManager.load_from_csv(
        raw, FILENAME, CHUNK_SIZE, DTYPES, converters=converters)
    # saved_raw_df: bool = PersistenceManager.save_to_csv(
    #     raw_dataframe, processed, RAW_FILENAME)
    missing_dataframe = data_analyze(raw_dataframe, DEPENDENT_COLUMN)
    preprocessed_dataframe = fill_and_update(raw_dataframe)
    # saved_preprocessed_df: bool = PersistenceManager.save_to_csv(
    #     preprocessed_dataframe, processed, PREPROCESSED_FILENAME)

    # Plots
    plot_distribution(preprocessed_dataframe['tenure'], color=COLORS[0])
    plot_distribution(preprocessed_dataframe['MonthlyCharges'],
                      color=COLORS[1])
    plot_distribution(preprocessed_dataframe['TotalCharges'], color=COLORS[2])
    plot_count(preprocessed_dataframe, DEPENDENT_COLUMN)

    encoded_dataframe: pd.DataFrame = one_hot_encoding(preprocessed_dataframe)
    # saved_encoded_df: bool = PersistenceManager.save_to_csv(
    #     encoded_dataframe, processed, PREPROCESSED_FILENAME)
    scaled_dataframe: pd.DataFrame = scaling(encoded_dataframe)
    # saved_scaled_df: bool = PersistenceManager.save_to_csv(
    #     scaled_dataframe, processed, SCALED_FILENAME)
    x_train, x_test, y_train, y_test = split_data(
        scaled_dataframe, DEPENDENT_COLUMN, TEST_SIZE, RANDOM_STATE)

    logistic_regression: LogisticRegression = train_model(x_train, y_train)
    confusion_matrix: np.ndarray = get_model_metrics(
        logistic_regression, x_test, y_test)
    plot_confusion_matrix(confusion_matrix)
    classification_report: str = model_report(
        y_test, logistic_regression.predict(x_test),
        [DEPENDENT_COLUMN, 'No ' + DEPENDENT_COLUMN])
    fpr, tpr, auc = roc_curve(logistic_regression, x_test, y_test)
    plot_roc_curve(fpr, tpr, auc)

    # Regularization
    lasso_regression: LogisticRegression = train_model(
        x_train, y_train, INVERSE_REGULARIZATION_STRENGTH, SOLVER, PENALTY)
    lasso_matrix: np.ndarray = get_model_metrics(
        lasso_regression, x_test, y_test)
    plot_confusion_matrix(lasso_matrix, 'lasso')
    plot_weights(
        lasso_regression, scaled_dataframe.drop(
            DEPENDENT_COLUMN, axis=1).columns.values, 'lasso')

    ridge_regression: LogisticRegression = train_model(
        x_train, y_train, INVERSE_REGULARIZATION_STRENGTH, SOLVER)
    ridge_matrix: np.ndarray = get_model_metrics(
        ridge_regression, x_test, y_test)
    plot_confusion_matrix(ridge_matrix, 'ridge')
    plot_weights(
        ridge_regression, scaled_dataframe.drop(
            DEPENDENT_COLUMN, axis=1).columns.values, 'ridge')

