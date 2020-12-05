import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

y = y.reshape(len(y), 1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
reg = SVR(kernel='rbf')  # gaussian radial basis function kernel
reg.fit(x, y)

output = sc_y.inverse_transform(reg.predict(sc_x.transform([[6.5]])))
print(output)

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(reg.predict(x)), color='blue')
plt.title('Experience vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, reg.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
