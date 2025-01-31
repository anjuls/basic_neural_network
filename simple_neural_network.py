import numpy as np

# Step function (activation function)
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron class
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.lr = lr  # Learning rate

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return step_function(z)

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                pred = self.predict(X[i])
                error = y[i] - pred
                total_error += abs(error)
                # Update weights and bias
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error
            print(f"Epoch {epoch+1}: Error = {total_error}")

# AND gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate output

# Initialize and train perceptron
perceptron = Perceptron(input_size=2)
perceptron.train(X, y, epochs=10)

# Testing trained perceptron
print("\nTesting Trained Perceptron:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Prediction: {perceptron.predict(X[i])}")

