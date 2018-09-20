import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X_train = data.loc[:, ['Bedrooms', 'Bathrooms', 'Size']].values
Y_train = data.Price.values.reshape(-1, 1)

# Feature scaling so that Gradient Descent works well
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
Y_train = sc.fit_transform(Y_train)

# Parameters
alpha = 0.01  # learning rate
iterations = 1000  #  iteration
m, n = X_train.shape  # sample count in training data

# bias_vector to add to X
bias_vector = np.ones((m, 1))
X_train = np.append(bias_vector, X_train, axis=1)

# TF Placeholders
X = tf.placeholder(tf.float64, name='X')
Y = tf.placeholder(tf.float64, name='Y')

# initial Theta variables
theta = tf.Variable(np.random.normal(size=n+1), name="theta")

# Hypothesis
pred = tf.multiply(X, theta)

# Cost Function
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*m)

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
print(f"Coefficients theta = {sess.run(theta)}")

# Predicted Values
Y_pred = X_train.dot(sess.run(theta))
Y_pred = Y_pred.reshape(-1, 1)


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

# Close TF session
sess.close()
