import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Data preprocessing
data = pd.read_csv("data.csv")

# Converting Pandas dataframe to numpy array
X = data.x.values.reshape(-1, 1)
Y = data.y.values.reshape(-1, 1)

# Add bias
m = X.shape[0]  # sample count
bias = np.ones((m, 1))
X_train = np.append(bias, X, axis=1)

# Add x^2 values
x2 = X**2
X_train = np.append(X_train, x2, axis=1)
Y_train = np.array(Y)

# Initial Variables
theta = np.array([[0, 0, 0]]).reshape(-1, 1)
alpha = 0.0001
iterations = 100000


# Cost function
def cost_function(X, Y, B):
    m = Y.shape[0]
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J


print("Initial cost", cost_function(X_train, Y_train, theta))


# Gradient Descent Algorithm
def gradient_descent(X, Y, theta, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(theta)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        theta = theta - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, theta)
        cost_history[iteration] = cost

    return theta, cost_history


# 100000 Iterations
newTheta, cost_history = gradient_descent(
    X_train, Y_train, theta, alpha, iterations)


print("New theta", newTheta)
print("Final Cost", cost_history[-1])

# To see wheather Gradient Descent decrease cost in each iteration
plt.plot(range(iterations), cost_history)
plt.title("Gradient Descent")
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

# Predicted Values
Y_pred = X_train.dot(newTheta)


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
print("RMSE = ", rmse(Y_train, Y_pred))
print("R2 Score = ", r2_score(Y_train, Y_pred))


# Visualization
# Ploting Regression Line
plt.plot(X, Y_pred, color='#c93e4e', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#54a774', label='Scatter Plot')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
