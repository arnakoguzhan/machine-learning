import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X = data.Size.values
Y = data.Price.values

m = X.shape[0]  #  number of samples

# Reshaping
X = np.reshape(X, (m, 1))
y = np.reshape(Y, (m, 1))


# Calculating coefficient
def normalEquation(X, Y):
    m = X.shape[0]  #  number of samples
    theta = []

    # bias_vector to add to X
    bias_vector = np.ones((m, 1))

    # combine bias and X
    X = np.append(bias_vector, X, axis=1)

    # Now do the math :)
    # Normal Equation formula is :
    # inverse(X^T * X) * X^T * y
    X_transpose = np.transpose(X)
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    return (theta[0], theta[1])


theta0, theta1 = normalEquation(X, Y)

# Printing coefficients
print(f"Coefficients theta0 = {theta0}, theta1 = {theta1} ")


# Predicted Values
Y_pred = theta0 + theta1 * X


# Model Evaluation
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)

print("MSE Score = ", mse)
print("RMSE Score = ", rmse)

# Calculating R2 Score
mean_y = np.mean(Y)

ss_tot = 0
ss_res = 0
for i in range(m):
    y_pred = theta0 + theta1 * X[i]
    ss_tot += (Y[i] - mean_y) ** 2
    ss_res += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
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
