import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network (Single-layer Perceptron)
class SpamDetector(nn.Module):
    def __init__(self):
        super(SpamDetector, self).__init__()
        self.layer = nn.Linear(3, 1)  # 3 input features -> 1 output

    def forward(self, x):
        return torch.sigmoid(self.layer(x))  # Sigmoid activation for binary classification

# Sample spam dataset (Feature Representation)
# Features: [Contains 'free', Contains 'win', Contains 'money']
X = torch.tensor([
    [1.0, 1.0, 1.0],  # "Free money! You win!" (Spam)
    [1.0, 0.0, 1.0],  # "Get free cash now" (Spam)
    [0.0, 1.0, 1.0],  # "Win big money today" (Spam)
    [1.0, 1.0, 0.0],  # "Free prize, you win!" (Spam)
    [0.0, 0.0, 0.0],  # "Hello, how are you?" (Not Spam)
    [0.0, 0.0, 1.0],  # "Let’s meet and discuss money" (Not Spam)
    [0.0, 1.0, 0.0],  # "Win a chance to travel" (Not Spam)
    [1.0, 0.0, 0.0],  # "This is a free resource" (Not Spam)
])
y = torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])  # Labels: Spam (1), Not Spam (0)

# Initialize model, loss function, and optimizer
model = SpamDetector()
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if epoch % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Test with new messages
print("\nTesting Model on New Messages:")
test_messages = torch.tensor([
    [1.0, 0.0, 1.0],  # "Free money for you!"
    [0.0, 1.0, 0.0],  # "Win a vacation now!"
    [0.0, 0.0, 0.0],  # "Hey, how's it going?"
])
with torch.no_grad():  # No need to compute gradients for inference
    predictions = model(test_messages).round()  # Round to 0 or 1
    for i, pred in enumerate(predictions):
        print(f"Message {i+1} -> {'Spam' if pred.item() == 1 else 'Not Spam'}")

