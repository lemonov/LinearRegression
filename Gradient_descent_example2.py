import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# J = w ** 2
# dJ = 2*w

def J(w1, w2):
    return (w1 ** 2) + (w2 ** 4)


def dJ_w1(w1):
    return 2 * w1


def dJ_w2(w2):
    return 4 * (w2 **3)


learning_rate = 0.01

size = 50

w1 = np.linspace(-20, 20, size)
w2 = np.linspace(-20, 20, size)

W1, W2 = np.meshgrid(w1, w2)

J_w = np.fromfunction(np.vectorize(lambda k, j: J(W1[k][j], W2[k][j])), (size, size), dtype=int)

print(J_w)

w1_n = 20
w2_n = 4

w1_arr = [w1_n]
w2_arr = [w2_n]

for i in range(0, 300, 1):
    w1_n = w1_n - (learning_rate * dJ_w1(w1_n))
    w2_n = w2_n - (learning_rate * dJ_w2(w2_n))
    print(w1_n)
    print(w2_n)
    w1_arr.append(w1_n)
    w2_arr.append(w2_n)

w1_arr = np.array(w1_arr)
w2_arr = np.array(w2_arr)

J_w_arr = J(w1_arr, w2_arr)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(w1_arr, w2_arr, J_w_arr, s=5)
ax.plot_surface(W1, W2, J_w, cmap=cm.coolwarm)
plt.show()
