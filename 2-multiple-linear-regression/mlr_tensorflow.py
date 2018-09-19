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
iteration = 1000  #  iteration
m, n = X_train.shape  # sample count in training data

# bias_vector to add to X
bias_vector = np.ones((m, 1))
X_train = np.append(bias_vector, X_train, axis=1)

# TF Placeholders
X = tf.placeholder(tf.float64, name='X')
Y = tf.placeholder(tf.float64, name='Y')

# initial Theta variables
theta = tf.Variable(np.random.normal(size=n+1), name="theta")

# Prediction formula
pred = tf.multiply(X, theta)

# Mean squared error and Gradient descent optimizer
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*m)

# Minimize minimizes W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

# Initializing the variables and creating session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Batch Gradient Descent
# feed data to algorithm
for epoch in range(iteration):
    for (x, y) in zip(X_train, Y_train):
        sess.run(optimizer, feed_dict={X: x, Y: y})

# Printing coefficients
print(f"Coefficients theta = {sess.run(theta)}")

# Predicted Values for plotting
Y_pred = X_train.dot(sess.run(theta))
m = Y_pred.shape[0]
Y_pred = np.reshape(Y_pred, (m, 1))


# Model Evaluation
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


mse = sess.run(cost, feed_dict={X: X_train, Y: Y_train})
print("MSE = ", mse)
print("R2 Score = ", r2_score(Y_train, Y_pred))

# Close TF session
sess.close()
