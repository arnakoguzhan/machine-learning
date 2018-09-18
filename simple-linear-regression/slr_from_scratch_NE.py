import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading Data
data = pd.read_csv('RealEstate.csv')
# print(data.head())

# Collecting X and Y
X = data.Size.values
Y = data.Price.values

# Calculating coefficient
m = len(X)


def normalEquation(X, Y, m):
    theta = []

    # bias_vector to add to X
    bias_vector = np.ones((m, 1))

    # We need to reshape original matrixs
    # so that we manipulate it with bias_vector
    X = np.reshape(X, (m, 1))
    y = np.reshape(Y, (m, 1))

    # combine bias and X
    X = np.append(bias_vector, X, axis=1)

    # Now do the math :)
    # Normal Equation formula is :
    # inverse(X^T * X) * X^T * y
    X_transpose = np.transpose(X)
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    return (theta[0], theta[1])


theta0, theta1 = normalEquation(X, Y, m)

# Printing coefficients
print(f"Coefficients theta0 = {theta0}, theta1 = {theta1} ")


# Prediction and
def predict(theta0, theta1, X):
    return theta0 + theta1 * X


# Predicted Values
Y_pred = predict(theta0, theta1, X)

# Model Evaluation
# Calculating Root Mean Squares Error
rmse = 0
for i in range(m):
    y_pred = theta0 + theta1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/m)
print("RMSE = ", rmse)

# Mean Y for calculate R2 score
mean_y = np.mean(Y)

# Calculating R2 Score
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
plt.plot(X, Y_pred, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
