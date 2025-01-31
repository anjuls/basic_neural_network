import torch
import torch.nn as nn
import torch.optim as optim

# Define the simplest neural network: a single-layer perceptron
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer = nn.Linear(2, 1)  # 2 input features -> 1 output

    def forward(self, x):
        return torch.sigmoid(self.layer(x))  # Sigmoid for binary classification

# AND gate dataset
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # Inputs
y = torch.tensor([[0.0], [0.0], [0.0], [1.0]])  # AND gate labels

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if epoch % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Test the trained model
print("\nTesting Trained Model:")
with torch.no_grad():  # No gradients needed for inference
    for i in range(len(X)):
        prediction = model(X[i]).round().item()
        print(f"Input: {X[i].tolist()} -> Prediction: {int(prediction)}")

