from random import randrange
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import pandas as pd
from math import sqrt


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            new_list = [float(i) for i in row]
            dataset.append(new_list)
    return dataset


# Split a dataset into a train and test set
def train_test_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


def estimate_coef(data):
    x = []
    y = []

    for item in data:
        x.append(item[0])
        y.append(item[1])

    x = np.array(x)
    y = np.array(y)

    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x - n*m_y*m_x)
    SS_xx = np.sum(x*x - n*m_x*m_x)

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return x, y, (b_0, b_1)


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1]*x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()


# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


# Simple linear regression algorithm
def predict(x, b0, b1):
    return b0 + b1 * x


def main():
    # dataset
    filename = 'satislar.csv'
    data = load_csv(filename)

    split = 0.6
    train, test = train_test_split(data, split)

    # estimating coefficients
    x, y, b = estimate_coef(train)

    actual = []
    pre = []
    x_test = []
    for item in test:
        actual.append(item[1])
        pre.append(predict(item[0], b[0], b[1]))
        x_test.append(item[0])

    RMSE = rmse_metric(actual, pre)

    print('RMSE: %.3f' % (RMSE))
    print("Estimated coefficients:\nb_0 = {}  \
         \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plt.scatter(x_test, pre)
    plt.scatter(x_test, actual)
    plot_regression_line(x, y, b)


if __name__ == "__main__":
    main()
