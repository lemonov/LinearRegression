import numpy as np
import random
import matplotlib.pyplot as plt
import math


def get_random(i, j):
    return random.uniform(0, 10)


def get_samples(n):
    x = np.fromfunction(np.vectorize(get_random), (n, 1), dtype=float)
    x = np.sort(x, 0)
    x = np.unique(x, axis=0)
    y = generator(x)
    return x, y


def generator(x):
    return np.sin(x * 2) / x / 100


def fit(x, y):
    return np.linalg.solve(x.T.dot(x), x.T.dot(y))


def plot_poly_with_samples(x_samples, y_samples):
    plt.scatter(x_samples, y_samples)
    plt.show()


def add_bias(x):
    bias = np.ones(x.shape)
    x = np.append(x, bias, axis=1)
    return x


def add_degree(x, degree):
    for deg in range(2, degree, 1):
        x_deg = x[:, 0] ** deg
        x_deg = np.reshape(x_deg, (x.shape[0], 1))
        x = np.append(x, x_deg, axis=1)
    return x


def r_sq(Y, Y_hat):
    SS_res = (Y - Y_hat).dot(Y - Y_hat)
    SS_tot = (Y - Y.mean()).dot(Y - Y.mean())
    r_squared = 1 - (SS_res / SS_tot)
    return r_squared


# main
(x, y) = get_samples(10)

min_x = min(x)
max_x = max(x)

for deg in range(2, 15, 1):
    # init
    curve_x = np.linspace(min_x - 1, max_x + 1, 100)
    curve_y = generator(curve_x)
    x_changed = x

    # show points and plot
    plt.scatter(x_changed, y)
    plt.plot(curve_x, curve_y)

    # add bias and degree columns
    x_changed = add_bias(x_changed)
    x_changed = add_degree(x_changed, deg)

    w = fit(x_changed, y)

    # calculate R_squared
    y_hat = x_changed.dot(w)
    y_hat = np.reshape(y_hat, (y_hat.shape[0],))
    y_reshaped = np.reshape(y, (y.shape[0],))
    print(r_sq(y_reshaped, y_hat))

    curve_x = np.reshape(curve_x, (curve_x.shape[0], 1))
    curve_x = add_bias(curve_x)
    curve_x = add_degree(curve_x, deg)

    y_hat = curve_x.dot(w)
    plt.plot(curve_x[:, 0], y_hat)
    plt.show(block = True)
