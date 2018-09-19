import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X = data.Size.values
Y = data.Price.values

m = X.shape[0]  #  number of samples

X = X.reshape((m, 1))

# Model Intialization
reg = LinearRegression()

# Data Fitting to LinearRegression Model
reg = reg.fit(X, Y)

# Printing coefficients
print(f"Coefficients theta0 = {reg.intercept_}, theta1 = {reg.coef_} ")

# Predicted Values
Y_pred = reg.predict(X)

# Model Evaluation
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2 = reg.score(X, Y)

print("MSE = ", mse)
print("RMSE = ", rmse)
print("R2 Score = ", r2)

# Visualization
# Ploting Line
plt.plot(X, Y_pred, color='#c93e4e', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#54a774', label='Scatter Plot')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
