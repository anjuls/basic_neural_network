import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample dataset (spam and non-spam messages)
spam_data = [
    ("Win a free iPhone now!", 1),
    ("Claim your free prize today!", 1),
    ("You have won $1000, claim now!", 1),
    ("Limited-time offer! Win big money!", 1),
    ("Meeting at 3 PM", 0),
    ("Let's discuss the project tomorrow", 0),
    ("Are you available for a call?", 0),
    ("This is an important business update", 0),
]

# Extract texts and labels
texts, labels = zip(*spam_data)

# Convert texts into TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()  # Convert to NumPy array
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # Convert labels to tensor

# Define a simple neural network for spam detection
class SpamDetector(nn.Module):
    def __init__(self, input_size):
        super(SpamDetector, self).__init__()
        self.layer = nn.Linear(input_size, 1)  # Input features -> 1 output

    def forward(self, x):
        return torch.sigmoid(self.layer(x))  # Sigmoid activation for binary classification

# Initialize the model
input_size = X.shape[1]  # Number of features from TF-IDF
model = SpamDetector(input_size)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Convert NumPy array to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    outputs = model(X_tensor)  # Forward pass
    loss = criterion(outputs, y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

# Function to predict spam from user input
def predict_spam(user_input):
    user_tfidf = vectorizer.transform([user_input]).toarray()  # Convert input to TF-IDF
    user_tensor = torch.tensor(user_tfidf, dtype=torch.float32)  # Convert to tensor
    
    with torch.no_grad():
        prediction = model(user_tensor).item()  # Get prediction
        return "🚨 SPAM 🚨" if prediction >= 0.5 else "✅ NOT Spam ✅"

# Interactive loop
print("\n💬 Spam Detector (TF-IDF Powered) is running... (Press Ctrl+C to exit)")
try:
    while True:
        user_input = input("\n📩 Enter a message: ")
        if not user_input.strip():
            print("⚠️ Please enter a valid message!")
            continue
        
        print("📢 Prediction:", predict_spam(user_input))
except KeyboardInterrupt:
    print("\n👋 Exiting Spam Detector. Have a great day!")

