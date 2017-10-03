import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def fit(X, Y):
    den = X.dot(X) - X.mean() * X.sum()
    a = (X.dot(Y) - Y.mean() * X.sum()) / den
    b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / den
    return a, b

def r_sq(Y, Y_hat):
    SS_res = (Y-Y_hat).dot(Y-Y_hat)
    SS_tot = (Y-Y.mean()).dot(Y-Y.mean())
    r_squared = 1 - (SS_res/SS_tot)
    return r_squared


data = pd.read_csv("data/moore_parsed.csv", sep=";", error_bad_lines=False)
data = data
data_array = data.as_matrix(columns=['Year', 'Transistors'])
data_array = data_array.T
X = np.array(data_array[0], dtype=np.float64)
Y = np.array(data_array[1], dtype=np.float64)

Y = np.log(Y)

(a, b) = fit(X, Y)
print(a, b)

y_hat = a * X + b

R_sq = r_sq(Y, y_hat)
print(R_sq)

plt.plot(X, y_hat)
plt.scatter(X, Y)
plt.show()

year = 2017
estimated_number_of_transistors = np.exp(b) + np.exp(a * year)
print(estimated_number_of_transistors)

