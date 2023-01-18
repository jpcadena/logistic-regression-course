"""
Binomial Logistic Regression.
"""
from typing import Optional

import numpy as np
from numpy import float16
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
DEPENDENT_COLUMN: str = 'Churn'
FILE_PATH: str = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
CHUNK_SIZE: int = 500


def data_loading() -> pd.DataFrame:
    dtypes: dict[str, str] = {
        'customerID': str, 'gender': 'category', 'SeniorCitizen': 'uint8',
        'Partner': 'category', 'Dependents': 'category', 'tenure': 'uint8',
        'PhoneService': 'category', 'MultipleLines': 'category',
        'InternetService': 'category', 'OnlineSecurity': 'category',
        'OnlineBackup': 'category', 'DeviceProtection': 'category',
        'TechSupport': 'category', 'StreamingTV': 'category',
        'StreamingMovies': 'category', 'Contract': 'category',
        'PaperlessBilling': 'category', 'PaymentMethod': 'category',
        'MonthlyCharges': 'float16', 'Churn': 'category'}
    text_file_reader = pd.read_csv(
        FILE_PATH, index_col=0, dtype=dtypes, chunksize=CHUNK_SIZE,
        converters={'TotalCharges': lambda x: float16(x.replace(' ', '0.0'))})
    dataframe: pd.DataFrame = pd.concat(text_file_reader, ignore_index=True)
    return dataframe


customers: pd.DataFrame = data_loading()


def dataframe_information(
        dataframe: pd.DataFrame, dependent_column: str) -> Optional[pd.Series]:
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


missing_dataframe = dataframe_information(customers, DEPENDENT_COLUMN)

# Modifying some columns
customers.loc[customers["TotalCharges"] == 0.0, "TotalCharges"] = customers[
    "MonthlyCharges"]
customers["Churn"] = np.where(customers["Churn"] == "Yes", 1, 0)
customers["Churn"] = customers["Churn"].astype('uint8')
processing_customers: pd.DataFrame = pd.get_dummies(customers, dtype='uint8')
print(processing_customers)

# Correlation
fig = plt.figure(figsize=(15, 10))
processing_customers.corr()['Churn'].sort_values(ascending=True).plot(
    kind='bar')
plt.show()

# Scaling
scaler: MinMaxScaler = MinMaxScaler()
processing_customers_scaled = scaler.fit_transform(processing_customers)
processing_customers_scaled: pd.DataFrame = pd.DataFrame(
    processing_customers_scaled)
processing_customers_scaled.columns = processing_customers.columns
print(processing_customers_scaled)

# Plots
sns.countplot(data=customers, x='gender', hue='Churn')
plt.show()


def plot_categorical_columns(column: str):
    plt.figure(figsize=(10, 10))
    sns.countplot(data=customers, x=column, hue='Churn')
    plt.show()


categorical_columns = customers.select_dtypes(include='category').columns
for _ in categorical_columns:
    plot_categorical_columns(_)
plt.figure(figsize=(10, 10))
sns.pairplot(data=customers, hue='Churn')
plt.show()

# Splitting
independent_variables = processing_customers_scaled.drop('Churn', axis=1)
dependent_variable = processing_customers_scaled['Churn'].values
x_train, x_test, y_train, y_test = train_test_split(
    independent_variables, dependent_variable, test_size=0.3, random_state=42)

# Modelling
logistic_regression: LogisticRegression = LogisticRegression(max_iter=10000)
logistic_regression.fit(x_train, y_train)

# Evaluation
print(logistic_regression.score(x_test, y_test))
cm = confusion_matrix(logistic_regression.predict(x_test), y_test)
sns.heatmap(
    cm,
    annot=True,
    cmap='gray',
    cbar=False,
    square=True,
    fmt="d"
)
plt.ylabel('Real Label')
plt.xlabel('Predicted Label')
plt.show()
print(logistic_regression.predict_proba(x_test))
print(logistic_regression.coef_)

weights: pd.Series = pd.Series(
    logistic_regression.coef_[0],
    index=independent_variables.columns.values).sort_values(ascending=False)
plt.figure(figsize=(15, 5))
weights.plot(kind='bar')
plt.show()

# Regularization
# L1 Lasso
lasso: LogisticRegression = LogisticRegression(
    max_iter=10000, penalty='l1', solver='saga', C=0.5)
lasso.fit(x_train, y_train)
print(lasso.score(x_test, y_test))
cm_lasso = confusion_matrix(lasso.predict(x_test), y_test)
sns.heatmap(
    cm_lasso,
    annot=True,
    cmap='gray',
    cbar=False,
    square=True,
    fmt="d"
)
plt.ylabel('Real Label')
plt.xlabel('Predicted Label')
plt.show()
weights_lasso: pd.Series = pd.Series(
    lasso.coef_[0],
    index=independent_variables.columns.values).sort_values(ascending=False)
plt.figure(figsize=(15, 5))
weights_lasso.plot(kind='bar')
plt.show()
print(weights_lasso[weights_lasso == 0])

# L2 Ridge
ridge: LogisticRegression = LogisticRegression(
    max_iter=10000, penalty='l2', solver='saga', C=0.5)
ridge.fit(x_train, y_train)
print(ridge.score(x_test, y_test))
cm_ridge = confusion_matrix(ridge.predict(x_test), y_test)
sns.heatmap(
    cm_ridge,
    annot=True,
    cmap='gray',
    cbar=False,
    square=True,
    fmt="d"
)
plt.ylabel('Real Label')
plt.xlabel('Predicted Label')
plt.show()
weights_ridge: pd.Series = pd.Series(
    ridge.coef_[0],
    index=independent_variables.columns.values).sort_values(ascending=False)
plt.figure(figsize=(15, 5))
weights_ridge.plot(kind='bar')
plt.show()

# ElasticNet
val_c: np.ndarray = np.arange(0, 1, 0.01)
acc: list = []
for i in val_c:
    acc.append(
        LogisticRegression(
            max_iter=10000, penalty='elasticnet', solver='saga',
            l1_ratio=i).fit(x_train, y_train).score(x_test, y_test))
plt.plot(val_c, acc)
plt.show()
