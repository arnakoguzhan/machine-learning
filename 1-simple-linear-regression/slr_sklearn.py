import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X = data.Size.values.reshape(-1, 1)
Y = data.Price.values.reshape(-1, 1)

m = X.shape[0]  #  number of samples

# Model Intialization
reg = LinearRegression()

# Data Fitting to LinearRegression Model
reg = reg.fit(X, Y)

# Printing coefficients
print(f"Coefficients theta0 = {reg.intercept_}, theta1 = {reg.coef_} ")

# Predicted Values
Y_pred = reg.predict(X)


# Model Evaluation
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / Y.shape[0])
    return rmse


def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


# Print Scores
print("RMSE = ", rmse(Y, Y_pred))
print("R2 Score = ", r2_score(Y, Y_pred))

# Visualization
# Ploting Line
plt.plot(X, Y_pred, color='#c93e4e', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#54a774', label='Scatter Plot')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
