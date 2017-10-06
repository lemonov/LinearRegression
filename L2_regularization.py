import numpy as np
import matplotlib.pyplot as plt

N = 50

x = np.linspace(0, 10, N)
y = 0.5 * x + np.random.randn(N)
y[-24] -= 30
y[-21] -= 30
y[-2] -= 30
y[-50] -= 30
y[-11] -= 30


x = np.vstack((x, np.ones((1, N)))).T
w_maximum_likelyhood = np.linalg.solve(x.T.dot(x), x.T.dot(y))

y_hat_maximum_likelyhood = x.dot(w_maximum_likelyhood)

plt.scatter(x[:, 0], y)
plt.plot(x[:, 0], y_hat_maximum_likelyhood)
plt.show()

l2 = 1000.0
w_l2 = np.linalg.solve(l2*np.eye(2) + x.T.dot(x), y.T.dot(x))

y_hat_l2 = x.dot(w_l2)

plt.scatter(x[:, 0], y)
plt.plot(x[:, 0], y_hat_l2)
plt.show()
