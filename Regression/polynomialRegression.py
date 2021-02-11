from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
"------IMPORTING LIBRARIES---------"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"----IMPORTING DATASET---------"
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

"--TRAINING THE LINEAR REGRESSION MODEL ON THE WHOLE DATASET--"
linReg = LinearRegression()
linReg.fit(x, y)

"--TRAINING THE POLYNOMIAL REGRESSION MODEL ON THE WHOLE DATASET--"
polyReg = PolynomialFeatures(degree=4)
x_poly = polyReg.fit_transform(x)
linRegNew = LinearRegression()
linRegNew.fit(x_poly, y)

"--VISUALISING THE LINEAR REGRESSION RESULTS--"
# plt.scatter(x, y, color="red")
# plt.plot(x, linReg.predict(x), color="blue")
# plt.title("Truth or Bluff (Linear Regression)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

"--VISUALIZING THE POLYNOMIAL REGRESSION RESULTS--"
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color="red")
plt.plot(x, linRegNew.predict(polyReg.fit_transform(x)), color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
# plt.show()

"--PREDICT A NEW RESULT WITH LINEAR REGRESSION--"
print(linReg.predict([[6.5]]))

"--PREDICT A NEW RESULT BASED WITH POLYNOMIAL REGRESSION--"
print(linRegNew.predict(polyReg.fit_transform([[6.5]])))
