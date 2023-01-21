"""
Config script
"""
import os
from dotenv import load_dotenv
from numpy import uint8, uint16, float16

dotenv_path: str = '.env'
load_dotenv(dotenv_path=dotenv_path)

MAX_COLUMNS: int = int(os.getenv('MAX_COLUMNS'))
WIDTH: int = int(os.getenv('WIDTH'))
FILENAME: str = os.getenv('FILENAME')
CHUNK_SIZE: uint16 = uint16(os.getenv('CHUNK_SIZE'))
RAW_FILENAME: str = os.getenv('RAW_FILENAME')
PREPROCESSED_FILENAME: str = os.getenv('PREPROCESSED_FILENAME')
ENCODED_FILENAME: str = os.getenv('ENCODED_FILENAME')
SCALED_FILENAME: str = os.getenv('SCALED_FILENAME')
DEPENDENT_COLUMN: str = os.getenv('DEPENDENT_COLUMN')
TEST_SIZE: float16 = float16(os.getenv('TEST_SIZE'))
RANDOM_STATE: uint16 = uint16(os.getenv('RANDOM_STATE'))
PENALTY: str = os.getenv('PENALTY')
INVERSE_REGULARIZATION_STRENGTH: float16 = float16(os.getenv(
    'INVERSE_REGULARIZATION_STRENGTH'))
SOLVER: str = os.getenv('SOLVER')
RE_PATTERN: str = os.getenv('RE_PATTERN')
RE_REPL: str = os.getenv('RE_REPL')
PALETTE: str = os.getenv('PALETTE')
FONT_SIZE: uint8 = uint8(os.getenv('FONT_SIZE'))
ENCODING: str = os.getenv('ENCODING')

FIG_SIZE: tuple[uint8, uint8] = (15, 8)
COLORS: list[str] = ['lightskyblue', 'coral', 'palegreen']
DTYPES: dict = {
    'customerID': str, 'gender': 'category', 'SeniorCitizen': 'uint8',
    'Partner': 'category', 'Dependents': 'category', 'tenure': 'uint8',
    'PhoneService': 'category', 'MultipleLines': 'category',
    'InternetService': 'category', 'OnlineSecurity': 'category',
    'OnlineBackup': 'category', 'DeviceProtection': 'category',
    'TechSupport': 'category', 'StreamingTV': 'category',
    'StreamingMovies': 'category', 'Contract': 'category',
    'PaperlessBilling': 'category', 'PaymentMethod': 'category',
    'MonthlyCharges': 'float16', 'Churn': 'category'}
converters: dict = {
    'TotalCharges': lambda x: float16(x.replace(' ', '0.0'))}
NUMERICS: list[str] = [
    'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32',
    'int64', 'float16', 'float32', 'float64']
RANGES: list[tuple] = [
    (0, 255), (0, 65535), (0, 4294967295), (0, 18446744073709551615),
    (-128, 127), (-32768, 32767), (-2147483648, 2147483647),
    (-18446744073709551616, 18446744073709551615)]
