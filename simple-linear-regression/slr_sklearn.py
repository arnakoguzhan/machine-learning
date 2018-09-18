import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Reading Data
data = pd.read_csv('RealEstate.csv')
# print(data.head())

# Collecting X and Y
X = data.Size.values
Y = data.Price.values

m = len(X)
X = X.reshape((m, 1))

# Model Intialization
reg = LinearRegression()
# Data Fitting to LinearRegression Model
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)

print("RMSE = ", rmse)
print("R2 Score = ", r2)

# Visualization
# Ploting Line
plt.plot(X, Y_pred, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.savefig("graph.png")
plt.show()
