"""
First Analysis script
"""
from typing import Optional
import pandas as pd


def data_analyze(
        dataframe: pd.DataFrame, dependent_column: str) -> Optional[pd.Series]:
    """
    First analysis of given dataframe with information about
     its column, data types and missing values
    :param dataframe: DataFrame to analyze
    :type dataframe: pd.DataFrame
    :param dependent_column: Column name to check imbalance
    :type dependent_column: str
    :return: Missing values series if exists
    :rtype: pd.Series
    """
    print(dataframe.head())
    print(dataframe.shape)
    print(dataframe.dtypes)
    print(dataframe.info(verbose=True, memory_usage="deep"))
    print(dataframe.describe(include='all'))

    # Identifying Class Imbalance
    print(dataframe[dependent_column].value_counts())
    print(dataframe[dependent_column].unique())
    print(dataframe[dependent_column].value_counts(normalize=True) * 100)

    # missing values
    missing_values: pd.Series = (dataframe.isnull().sum())
    if len(missing_values) > 0:
        print(missing_values[missing_values > 0])
        print(missing_values[missing_values > 0] / dataframe.shape[0] * 100)
        return missing_values[missing_values > 0]
    return None
