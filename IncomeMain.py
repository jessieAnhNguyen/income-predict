from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

#Read the data from the file
df = pd.read_excel(r'C:\Users\Phuong Anh\tutorial-env\PycharmProjects\IncomePredict\venv\Scripts\AdultData.xlsx')
df.drop('Column1', 1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#Data preprocessing -- missing values
imputer = Imputer('NaN', 'mean', 1)
imputer = imputer.fit(X)
X = imputer.transform(X)


#Plot the graph

plt.figure()
plt.scatter(X.iloc[:, 1], y)
plt.show()


more = df.loc[y == 1]
less = df.loc[y == 0]

#Plot the data
X_num1 = range(len(more))
X_num2 = range(len(less))
plt.scatter(X_num1, more)
plt.scatter(X_num2, less)
plt.show()

plt.scatter(more.iloc[:, 0], more.iloc[:, 1])
plt.scatter(less.iloc[:, 0], less.iloc[:, 1])
plt.show()


#Logistic Regression using sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()

model = model.fit(X_train, y_train)
predicted_classes = model.predict(X)
parameters = model.coef_

print('Logistic regression score for training set: ', model.score(X_train, y_train))
print('Parameters: ', parameters)

y_true, y_pred = y_test, model.predict(X_test)
print(classification_report(y_true, y_pred))
