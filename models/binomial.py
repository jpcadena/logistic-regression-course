"""
Binomial Logistic Regression.
"""
import numpy as np
from numpy import float16
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

file_path: str = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
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
reader = pd.read_csv(
    file_path, index_col=0, dtype=dtypes, chunksize=500,
    converters={'TotalCharges': lambda x: float16(x.replace(' ', '0.0'))})
customers: pd.DataFrame = pd.concat(reader, ignore_index=True)

# Exploratory Data Analysis
print(customers.head())
print(customers.shape)
print(customers.dtypes)
print(customers.info(verbose=True, memory_usage="deep"))
print(customers.describe(include='all'))
missing_values = (customers.isnull().sum())
print(missing_values[missing_values > 0])
print(missing_values[missing_values > 0] / customers.shape[0] * 100)

# Identifying Class Imbalance
print(customers['Churn'].value_counts())
print(customers['Churn'].unique())
print(customers['Churn'].value_counts(normalize=True) * 100)

# Modifying some columns
customers.loc[customers["TotalCharges"] == 0.0, "TotalCharges"] = customers[
    "MonthlyCharges"]
# customers['TotalCharges'] = np.where(
#     customers["TotalCharges"] == 0.0, customers["MonthlyCharges"],
#     customers["TotalCharges"])
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


def plot_categorical_columns(column):
    plt.figure(figsize=(10, 10))
    sns.countplot(data=customers, x=column, hue='Churn')
    plt.show()


categorical_columns = customers.select_dtypes(include='category').columns

for _ in categorical_columns:
    plot_categorical_columns(_)

plt.figure(figsize=(10, 10))
sns.pairplot(data=customers, hue='Churn')
plt.show()

# Modelling
