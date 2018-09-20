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

# Feature scaling so that Gradient Descent works well
sc = StandardScaler()
X = sc.fit_transform(X)
Y = sc.fit_transform(Y)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=0)

# number of samples and features
m, n = X_train.shape

# bias_vector to add to X
bias_vector = np.ones((m, 1))
X_train = np.append(bias_vector, X_train, axis=1)
X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)

# Initial Coefficients
theta = np.random.normal(size=n+1).reshape(-1, 1)
Y_train = np.array(Y_train)


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


# Initial variables
alpha = 0.001  #  Learning rate
iteration = 100000  #  iteration

# Calculate new theta with gradient descent
newTheta, cost_history = gradient_descent(
    X_train, Y_train, theta, alpha, iteration)

print("New theta", newTheta)
print("Final Cost", cost_history[-1])

# To see wheather Gradient Descent decrease cost in each iteration
plt.plot(range(iteration), cost_history)
plt.title("Gradient Descent")
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

# Predicted Values
Y_pred = X_test.dot(newTheta)


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


# Reverse Feature Scaling so that graph will result in real values
X_test = sc.inverse_transform(X_test)
Y_pred = sc.inverse_transform(Y_pred)
Y_test = sc.inverse_transform(Y_test)
X = sc.inverse_transform(X)
Y = sc.inverse_transform(Y)

# Visualization
# Ploting Regression Line
plt.plot(X_test[:, 1:], Y_pred, color='#c93e4e', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#54a774', label='Scatter Plot')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
