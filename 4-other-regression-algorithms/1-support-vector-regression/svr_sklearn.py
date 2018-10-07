import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Data preprocessing
data = pd.read_csv("data.csv")

# Converting Pandas dataframe to numpy array
X = data.x.values.reshape(-1, 1)
Y = data.y.values.reshape(-1, 1)

# Feature scaling so that Gradient Descent works well
sc = StandardScaler()

X = sc.fit_transform(X)
Y = sc.fit_transform(Y)

# Model Intialization (RBF kernel)
reg = SVR(kernel='rbf')
# Data Fitting to SVR Model
reg = reg.fit(X, Y)
# Predicted Values
Y_pred = reg.predict(X)

# kernel can be 'linear', 'poly', 'rbf', 'sigmoid'
# Model Intialization (POLY kernel)
reg_poly = SVR(kernel='poly')
reg_poly = reg_poly.fit(X, Y)
Y_pred_poly = reg_poly.predict(X)

# Model Evaluation
r2 = reg.score(X, Y)  #  R2 Score
print("RBF R2 Score = ", r2)

r2 = reg_poly.score(X, Y)  #  R2 Score
print("POLY R2 Score = ", r2)

# Visualization
# Ploting Line
plt.plot(X, Y_pred, color='#c93e4e', label='Regression Line RBF')
plt.plot(X, Y_pred_poly, color='blue', label='Regression Line POLY')
# Ploting Scatter Points
plt.scatter(X, Y, c='#54a774', label='Scatter Plot')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
