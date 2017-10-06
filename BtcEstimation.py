import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    global X, Y
    data_csv = pd.read_csv("data/btc_data.csv")
    data_csv = data_csv.dropna(subset=['Open'])
    data = data_csv[['Timestamp', 'Weighted_Price']]
    X = data['Timestamp']
    Y = data['Weighted_Price']
    X = X.as_matrix()
    Y = Y.as_matrix()
    N = X.shape[0]
    X = np.reshape(X, (N, 1))
    Y = np.reshape(Y, (N, 1))
    return X, Y

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
    Y = np.reshape(Y, (Y.shape[0],))
    Y_hat = np.reshape(Y_hat, (Y_hat.shape[0],))
    SS_res = (Y - Y_hat).dot(Y - Y_hat)
    SS_tot = (Y - Y.mean()).dot(Y - Y.mean())
    r_squared = 1 - (SS_res / SS_tot)
    return r_squared


(X, Y) = load_data()

R_2 = []
R_2_L1 = []
R_2_L2 = []

MAX_DEG = 9
# L1 = 100
L2 = 1000

BEST_R = 0
BEST_W = 0
BEST_DEG = 0

for deg in range(2, MAX_DEG, 1):
    X_train = add_bias(X)
    X_train = add_degree(X_train, deg)

    W = np.linalg.solve(X_train.T.dot(X_train), X_train.T.dot(Y))
    # W_L1 = np.linalg.solve(L1*np.sign(X_train.shape[1]) + X_train.T.dot(X_train), X_train.T.dot(Y))
    W_L2 = np.linalg.solve(L2*np.eye(X_train.shape[1]) + X_train.T.dot(X_train), X_train.T.dot(Y))

    Y_hat = X_train.dot(W)
    # Y_hat_L1 = X_train.dot(W_L1)
    Y_hat_L2 = X_train.dot(W_L2)

    r_squared = r_sq(Y, Y_hat)
    if r_squared > BEST_R:
        BEST_R = r_squared
        BEST_W = W
        BEST_DEG = deg

    # r_squared_l1 = r_sq(Y, Y_hat_L1)
    r_squared_l2 = r_sq(Y, Y_hat_L2)
    if r_squared_l2 > BEST_R:
        BEST_R = r_squared_l2
        BEST_W = W_L2
        BEST_DEG = deg

    R_2.append(r_squared)
    # R_2_L1.append(r_squared_l1)
    R_2_L2.append(r_squared_l2)

    # plt.plot(X_train[:, 0], Y_hat, label="Pure poly regression")
    # # plt.plot(X_train[:, 0], Y_hat_L1, label="With L1 regularization")
    # plt.plot(X_train[:, 0], Y_hat_L2, label="With L2 regularization")
    # plt.legend()
    # plt.scatter(X_train[:, 0], Y, s=1)
    # plt.show()

print(BEST_W)
print(BEST_R)
deg_x = np.linspace(2, MAX_DEG, MAX_DEG-2)
plt.xlabel("Degree")
plt.ylabel("R^2")
plt.plot(deg_x, R_2, label="Pure poly regression")
# plt.plot(deg_x, R_2_L1, label="With L1 regularization")
plt.plot(deg_x, R_2_L2, label="With L2 regularization")
plt.legend()
plt.show()

#
# learning_speed = 0.000001
# T = 15
# # w = BEST_W
# X_train = add_bias(X)
# # X_train = add_degree(X_train, BEST_DEG)
# w = np.random.randn(2,1)
#
# for i in range(0, T, 1):
#     print("Descent :"+str(i))
#     Y_hat = X_train.dot(w)
#     w = w - learning_speed * (X_train.T.dot(Y_hat - Y))
#     print(w)
#
# Y_HAT_GD = X_train.dot(w)
#
# print(w)
# plt.scatter(X_train[:, 0], Y, s=1)
# plt.plot(X_train[:, 0], Y_HAT_GD)
# plt.show()




