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

# Function to convert user input into feature vector
def text_to_features(text):
    text = text.lower()
    return torch.tensor([[float('free' in text), float('win' in text), float('money' in text)]])

# Interactive loop for spam detection
print("\n💬 Spam Detector is running... (Press Ctrl+C to exit)")
try:
    while True:
        user_input = input("\n📩 Enter a message: ")
        if not user_input.strip():
            print("⚠️ Please enter a valid message!")
            continue
        
        # Convert input to features
        features = text_to_features(user_input)
        
        # Predict and display result
        with torch.no_grad():
            prediction = model(features).round().item()
            print("📢 Prediction:", "🚨 SPAM 🚨" if prediction == 1 else "✅ NOT Spam ✅")
except KeyboardInterrupt:
    print("\n👋 Exiting Spam Detector. Have a great day!")

