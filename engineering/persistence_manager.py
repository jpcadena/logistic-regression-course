"""
Persistence script
"""
from enum import Enum
import pandas as pd
from pandas.io.parsers import TextFileReader
from core.config import ENCODING, CHUNK_SIZE


class DataPath(str, Enum):
    """
    Data Type class based on Enum
    """
    RAW: str = 'data/raw/'
    PROCESSED: str = 'data/processed/'
    FIGURES: str = 'reports/figures/'


class PersistenceManager:
    """
    Persistence Manager class
    """

    @staticmethod
    def save_to_csv(
            data: list[dict] | pd.DataFrame, path: DataPath,
            filename: str) -> bool:
        """
        Save list of dictionaries as csv file
        :param data: list of tweets as dictionaries
        :type data: list[dict]
        :param path: folder where data will be saved to
        :type path: DataPath
        :param filename: name of the file
        :type filename: str
        :return: confirmation for csv file created
        :rtype: bool
        """
        dataframe: pd.DataFrame
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            if not data:
                return False
            dataframe = pd.DataFrame(data)
        path_or_buf: str = path.value + filename
        dataframe.to_csv(path_or_buf, index=False, encoding=ENCODING)
        return True

    @staticmethod
    def load_from_csv(
            path: DataPath, filename: str, chunk_size: int = CHUNK_SIZE,
            dtypes: dict = None,
            parse_dates: list[str] = None,
            converters: dict = None
    ) -> pd.DataFrame:
        """
        Load dataframe from CSV using chunk scheme
        :param path: Path for data
        :type path: DataPath
        :param filename: name of the file
        :type filename: str
        :param chunk_size: Number of chunks to split dataset
        :type chunk_size: int
        :param parse_dates: List of dataframe date columns to parse
        :type parse_dates: list[str]
        :return: dataframe retrieved from CSV after optimization with chunks
        :rtype: pd.DataFrame
        """
        filepath: str = path.value + filename
        text_file_reader: TextFileReader = pd.read_csv(
            filepath, index_col=0, chunksize=chunk_size, encoding=ENCODING,
            dtype=dtypes,
            parse_dates=parse_dates,
            converters=converters
        )
        dataframe: pd.DataFrame = pd.concat(
            text_file_reader, ignore_index=True)
        return dataframe

    @staticmethod
    def save_to_pickle(
            dataframe: pd.DataFrame, path: DataPath, filename: str) -> None:
        """
        Save dataframe to pickle file
        :param dataframe: dataframe
        :type dataframe: pd.DataFrame
        :param filename: name of the pkl file
        :type filename: str
        :return: None
        :rtype: NoneType
        """
        dataframe.to_pickle(f'{path.value}{filename}')

    @staticmethod
    def load_from_pickle(path: DataPath, filename: str) -> pd.DataFrame:
        """
        Load dataframe from Pickle file
        :param filename: name of the pkl file to search and load
        :type filename: str
        :return: dataframe read from pickle
        :rtype: pd.DataFrame
        """
        dataframe: pd.DataFrame = pd.read_pickle(f'{path.value}{filename}')
        return dataframe
