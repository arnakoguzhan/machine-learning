import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X = data.Size.values
Y = data.Price.values

m = X.shape[0]  #  number of samples

# Reshaping so that feature scaling function works
X = np.reshape(X, (m, 1))
Y = np.reshape(Y, (m, 1))

# Feature scaling so that Gradient Descent works well
sc = StandardScaler()
X = sc.fit_transform(X)
Y = sc.fit_transform(Y)


def gradient_descent(alpha, X, Y, iterations):
    m = X.shape[0]  #  number of samples

    # bias_vector to add to X
    bias_vector = np.ones((m, 1))

    # combine bias and X
    X = np.append(bias_vector, X, axis=1)
    Y = np.reshape(Y, (m,))

    # theta
    theta = np.ones(2)

    cost_history = []

    X_transpose = X.transpose()
    for iter in range(iterations):
        hypothesis = np.dot(X, theta)
        loss = hypothesis - Y
        J = np.sum(loss ** 2) / (2 * m)  # cost

        cost_history.append(J)

        gradient = np.dot(X_transpose, loss) / m
        theta = theta - alpha * gradient  # update theta simultenously

    plt.plot(range(iterations), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    return (theta, J)


# Finding coefficients using Batch Gradient Descent algorithm
alpha = 0.01  # learning rate
theta, cost = gradient_descent(alpha, X, Y, 1000)

# Printing coefficients
print(f"Coefficients theta0 = {theta[0]}, theta1 = {theta[1]} ")

# Predicted Values
Y_pred = theta[0] + theta[1] * X


# Model Evaluation
m = X.shape[0]  #  sample count
rmse = np.sqrt(cost/m)
print("MSE = ", cost)
print("RMSE = ", rmse)

# Calculating R2 Score
mean_y = np.mean(Y)

ss_tot = 0
ss_res = 0
for i in range(m):
    y_pred = theta[0] + theta[1] * X[i]
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
