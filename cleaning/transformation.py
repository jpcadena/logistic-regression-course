"""
Main script
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def one_hot_encoding(dataframe: pd.DataFrame) -> pd.DataFrame:
    encoded_dataframe: pd.DataFrame = pd.get_dummies(dataframe, dtype='uint8')
    return encoded_dataframe


def scaling(dataframe: pd.DataFrame) -> pd.DataFrame:
    scaler: MinMaxScaler = MinMaxScaler()
    scaled_dataframe = scaler.fit_transform(dataframe)
    scaled_dataframe: pd.DataFrame = pd.DataFrame(
        scaled_dataframe)
    scaled_dataframe.columns = dataframe.columns
    return scaled_dataframe
