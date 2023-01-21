"""
Simple Logistic Regression.
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

digits: Bunch = load_digits()
print(digits.data[0])
image: np.ndarray = np.reshape(digits.data[10], (8, 8))
plt.imshow(image, cmap='gray')
print(digits.target[10])

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=0)

# Model
logistic_regression: LogisticRegression = LogisticRegression()
logistic_regression.fit(x_train, y_train)

# Evaluation
predictions: np.ndarray = logistic_regression.predict(x_test)
cm: np.ndarray = confusion_matrix(y_test, predictions)
print(cm)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, linewidths=.5, square=True, cmap='coolwarm')
plt.ylabel('actual label')
plt.xlabel('Predicted label')
plt.show()
