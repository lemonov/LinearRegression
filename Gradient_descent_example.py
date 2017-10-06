import numpy as np
import matplotlib.pyplot as plt


# J = w ** 2
# dJ = 2*w

def J(w):
    return w**2


def dJ(w):
    return 2 * w


initial_w = 20
learning_rate = 0.1
w = initial_w

w_arr = [w]

for i in range(0, 1000, 1):
    w = w - (dJ(w) * learning_rate)
    w_arr.append(w)
    print(w)

w_arr = np.array(w_arr)
J_w_arr = J(w_arr)

w = np.linspace(-20, 20, 100)
J_w = J(w)

plt.scatter(w_arr, J_w_arr)
plt.plot(w, J_w)
plt.show()
