import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X_train = data.Size.values
Y_train = data.Price.values

# Reshaping
X_train = np.reshape(X_train, (len(X_train), 1))
Y_train = np.reshape(Y_train, (len(Y_train), 1))

# Feature scaling so that Gradient Descent works well
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
Y_train = sc.fit_transform(Y_train)

# Parameters
alpha = 0.01  # learning rate
iteration = 1000  #  iteration
m = X_train.shape[0]  # sample count in training data

# TF Placeholders
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Theta variables
theta0 = tf.Variable(np.random.normal(), name="theta0")
theta1 = tf.Variable(np.random.normal(), name="theta1")

# Prediction formula
pred = tf.add(tf.multiply(X, theta1), theta0)

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
print(f"Coefficients theta0 = {sess.run(theta0)}, theta1 = {sess.run(theta1)}")

# Predicted Values for plotting
Y_pred = sess.run(theta0) + sess.run(theta1) * X_train


# Model Evaluation
mse = sess.run(cost, feed_dict={X: X_train, Y: Y_train})
print("MSE = ", mse)


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
