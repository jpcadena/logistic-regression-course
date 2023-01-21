"""
Preprocessing section including: Formatting, Cleaning
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import float16, uint16
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from core.config import NUMERICS, RANGES


def downcast_type(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Optimization of numeric columns by down-casting its datatype
    :param dataframe: dataframe to optimize
    :type dataframe: pd.DataFrame
    :return: optimized dataframe
    :rtype: pd.DataFrame
    """
    numerics: list[str] = NUMERICS[:-3]
    numeric_ranges: list[tuple] = RANGES
    df_num_cols: pd.DataFrame = dataframe.select_dtypes(include=numerics)
    for column in df_num_cols:
        new_type: str = numerics[numeric_ranges.index(
            [num_range for num_range in numeric_ranges if
             df_num_cols[column].min() > num_range[0] and
             num_range[1] <= df_num_cols[column].max()][0])]
        df_num_cols[column] = df_num_cols[column].apply(
            pd.to_numeric, downcast=new_type)  # check map for Pd.Series
    dataframe[df_num_cols.columns] = df_num_cols
    return dataframe


def fill_and_update(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe.loc[
        dataframe["TotalCharges"] == 0.0, "TotalCharges"] = dataframe[
        "MonthlyCharges"]
    dataframe["Churn"] = np.where(dataframe["Churn"] == "Yes", 1, 0)
    dataframe["Churn"] = dataframe["Churn"].astype('uint8')
    return dataframe


def lof_observation(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This function identifies outliers with LOF method
    :param dataframe: Dataframe containing data
    :type dataframe: pd.DataFrame
    :return: clean dataframe without outliers from LOF
    :rtype: pd.DataFrame
    """
    df_num_cols: pd.DataFrame = dataframe.select_dtypes(include=NUMERICS)
    df_outlier: pd.DataFrame = df_num_cols.astype("float64")
    clf: LocalOutlierFactor = LocalOutlierFactor(
        n_neighbors=20, contamination=0.1)
    clf.fit_predict(df_outlier)
    df_scores = clf.negative_outlier_factor_
    scores_df: pd.DataFrame = pd.DataFrame(np.sort(df_scores))
    scores_df.plot(stacked=True, xlim=[0, 20], color='r',
                   title='Visualization of outliers according to the LOF '
                         'method', style='.-')
    plt.savefig('reports/figures/outliers.png')
    plt.show()
    th_val = np.sort(df_scores)[2]
    outliers: bool = df_scores > th_val
    dataframe: pd.DataFrame = dataframe.drop(df_outlier[~outliers].index)
    print(dataframe.shape)
    return dataframe


def clear_outliers(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This function remove the outliers from specific column
    :param dataframe: Dataframe to clear out
    :type dataframe: pd.DataFrame
    :param column: Column name
    :type column: str
    :return: clean dataframe from outliers using IQR
    :rtype: pd.DataFrame
    """
    first_quartile: float = dataframe[column].quantile(0.25)
    third_quartile: float = dataframe[column].quantile(0.75)
    iqr: float = third_quartile - first_quartile
    lower: float = first_quartile - 1.5 * iqr
    upper: float = third_quartile + 1.5 * iqr
    print(f"{column}- Lower score: ", lower, "and upper score: ", upper)
    df_outlier = dataframe[column][(dataframe[column] > upper)]
    print(df_outlier)
    return dataframe


def split_data(
        dataframe: pd.DataFrame, dependent_col: str, test_size: float16,
        random_state: uint16
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    independent_variables = dataframe.drop(dependent_col, axis=1).to_numpy()
    dependent_variable = dataframe[dependent_col].values
    x_train, x_test, y_train, y_test = train_test_split(
        independent_variables, dependent_variable, test_size=test_size,
        random_state=random_state)
    print(type(y_train), "y_train")
    print(y_train)
    return x_train, x_test, y_train, y_test
