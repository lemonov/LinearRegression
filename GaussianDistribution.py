import numpy as np
import math
import matplotlib.pyplot as plt


def gaussian_distribution_probability_value(x, mean, deviaton):
    m = 1 / (math.sqrt(2 * math.pi * (deviaton ** 2)))
    e = -((x - mean) ** 2) / (2 * (deviaton ** 2))
    e = math.exp(e)
    return m * e


def gaussian_distribution_probability_vector(x, mean, deviation):
    y = np.array(x.shape)
    y = np.fromfunction(np.vectorize(lambda x_par: gaussian_distribution_probability_value(x_par, mean, deviation)),
                        x.shape)
    return y


def gaussian_value(x, a, b, c):
    e = math.exp((-0.5)*(((x-b)**2)/(c**2)))
    return a * e


def gaussian_vector(x, a, b, c):
    y = np.array(x.shape)
    y = np.fromfunction(np.vectorize(lambda x_par: gaussian_value(x_par, a, b, c)), x.shape)
    return y

print(math.log(math.exp(100)))
x = np.linspace(-10, 10, 100)
y = gaussian_vector(x, 2, 22, 10)
plt.hist(y)
plt.plot(x, y)
plt.show()
