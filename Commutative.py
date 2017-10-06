import numpy as np

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

print(A.dot(B))
print(A.T.dot(B))
print(A.T.dot(B.T))
print(A.dot(B.T))

