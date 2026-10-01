import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,1])

weights = np.random.rand(2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for epoch in range(1000):
    for inputs, target in zip(X, y):
        output = sigmoid(np.dot(inputs, weights))
        error = target - output
        weights += 0.1 * error * inputs

print("Hasil OR:")
for inputs in X:
    output = sigmoid(np.dot(inputs, weights))
    print(inputs, "=>", round(output))
