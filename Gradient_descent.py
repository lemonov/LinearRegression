import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 10
D = 3
X = np.zeros((N, D))
X[:, 0] = 1  # bias
X[:5, 1] = 1
X[5:, 2] = 1

Y = np.array([0]*5 + [1]*5)
print(X)
print(Y)

costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001

mse = []
for t in range(0, 1000, 1):
    Y_hat = X.dot(w)
    delta = Y_hat - Y
    w = w - learning_rate*X.T.dot(delta)
    mse.append(delta.dot(delta) / N)


plt.plot(mse)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X[:,1], X[:,2], Y, s=5)

ax.plot(X[:,1], X[:,2], Y_hat)

plt.show()