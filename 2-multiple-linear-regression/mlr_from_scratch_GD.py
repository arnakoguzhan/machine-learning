import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X = data.loc[:, ['Bedrooms', 'Bathrooms', 'Size']].values
Y = data.Price.values

m = X.shape[0]  #  number of samples

# Reshaping so that feature scaling function works
Y = np.reshape(Y, (m, 1))

# Feature scaling so that Gradient Descent works well
sc = StandardScaler()
X = sc.fit_transform(X)
Y = sc.fit_transform(Y)

# bias_vector to add to X
bias_vector = np.ones((m, 1))
X = np.append(bias_vector, X, axis=1)
Y = np.reshape(Y, (m,))


def gradient_descent(alpha, X, Y, iterations):
    m, n = X.shape  #  number of samples

    # theta
    theta = np.ones(n)

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
print(f"Coefficients theta = {theta} ")

# Predicted Values
Y_pred = X.dot(theta)

# List 10 predict and their actual values
# Remember values were scaled with StandardScaler !
for i in range(10):
    predict = Y_pred[i]
    actual = Y[i]
    print(f"predict = {predict}, actual = {actual}")


def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse


def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


print("RMSE = ", rmse(Y, Y_pred))
print("R2 Score = ", r2_score(Y, Y_pred))
