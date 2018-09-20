import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X = data.Size.values.reshape(-1, 1)
Y = data.Price.values.reshape(-1, 1)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=0)

# number of samples and features
m, n = X_train.shape

# bias_vector to add to X
bias_vector = np.ones((m, 1))
X_train = np.append(bias_vector, X_train, axis=1)
X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)


# Calculating theta with Normal Equation formula
def normalEquation(X, Y):
    theta = []

    # Normal Equation formula is :
    # inverse(X^T * X) * X^T * y
    X_transpose = np.transpose(X)
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(Y)

    return theta


# Get theta
theta = normalEquation(X_train, Y_train)

# Printing coefficients
print("Theta ", theta)

# Predicted Values
Y_pred = X_test.dot(theta)


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
print("RMSE = ", rmse(Y_test, Y_pred))
print("R2 Score = ", r2_score(Y_test, Y_pred))


# Visualization
# Ploting Regression Line
plt.plot(X_test[:, 1:], Y_pred, color='#c93e4e', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#54a774', label='Scatter Plot')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
