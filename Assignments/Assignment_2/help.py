import numpy as np

b0 = -3
b1 = 0.5
b2 = 0.5

def logistic(x1, x2):
    return np.exp(b0 + x1*b1 + x2*b2) / (1 + np.exp(b0 + x1*b1 + x2*b2))

data = [
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
    (5.0, 6.0),
    (6.0, 7.0),
    (7.0, 8.0),
    (8.0, 9.0)
]

for x1, x2 in data:
    print(f'x1 = {x1}, x2 = {x2}, pred = {np.round(logistic(x1, x2), 3)}\n')


import numpy as np

# Data points for each class
class_0 = np.array([
    [1, 1],
    [2, 1],
    [3, 2],
    [2, 3],
    [1, 3]
])

class_1 = np.array([
    [7, 1],
    [5, 2],
    [6, 4],
    [4, 5],
    [6, 5]
])

# Calculate the mean for each class
mean_0 = np.mean(class_0, axis=0)
mean_1 = np.mean(class_1, axis=0)

# Calculate the covariance matrix for each class
covariance_0 = np.cov(class_0, rowvar=False)
covariance_1 = np.cov(class_1, rowvar=False)

# Print results
print("Class 0 Mean Vector:")
print(mean_0)
print("\nClass 0 Covariance Matrix:")
print(covariance_0)

print("\nClass 1 Mean Vector:")
print(mean_1)
print("\nClass 1 Covariance Matrix:")
print(covariance_1)
