import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X_train = data.Size.values.reshape(-1, 1)
Y_train = data.Price.values.reshape(-1, 1)

# Feature scaling so that Gradient Descent works well
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
Y_train = sc.fit_transform(Y_train)

# Parameters
alpha = 0.01  # learning rate
iterations = 1000  #  number of iteration
m = X_train.shape[0]  # sample count in training data

# TF Placeholders
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Theta variables
theta0 = tf.Variable(np.random.normal(), name="theta0")
theta1 = tf.Variable(np.random.normal(), name="theta1")

# Hypothesis
hypothesis = tf.add(tf.multiply(X, theta1), theta0)

# Cost Function
cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2))/(2*m)

# Gradient Descent Algorithm
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

# Initializing the variables and creating session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Batch Gradient Descent
# feed data to algorithm
for iter in range(iterations):
    for (x, y) in zip(X_train, Y_train):
        sess.run(optimizer, feed_dict={X: x, Y: y})


# Printing coefficients
print(f"Coefficients theta0 = {sess.run(theta0)}, theta1 = {sess.run(theta1)}")


# Predicted Values for plotting
Y_pred = sess.run(theta0) + sess.run(theta1) * X_train


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


# Reverse Feature Scaling so that graph will result in real values
X_train = sc.inverse_transform(X_train)
Y_train = sc.inverse_transform(Y_train)
Y_pred = sc.inverse_transform(Y_pred)


# Visualization
# Ploting Line
plt.plot(X_train, Y_pred, color='#c93e4e', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X_train, Y_train, c='#54a774', label='Scatter Plot')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()

# Close TF session
sess.close()
