"""
Train QBNN for Japanese chatbot response classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from apqb_qbnn_v2 import QBNNLayerV2
from chatbot_dataset import CONVERSATIONS, RESPONSE_CATEGORIES
import pickle

def create_vectorizer():
    """Create TF-IDF vectorizer for Japanese text."""
    questions = [q for q, _, _ in CONVERSATIONS]
    vectorizer = TfidfVectorizer(
        max_features=128,
        ngram_range=(1, 2),
        analyzer='char',  # Character-level for Japanese
        lowercase=True
    )
    vectorizer.fit(questions)
    return vectorizer

class ChatbotQBNN(nn.Module):
    """QBNN for chatbot response classification."""
    def __init__(self, input_dim, hidden_dim=64, n_classes=12):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        in_dim = input_dim
        for i in range(2):
            self.layers.append(QBNNLayerV2(
                in_dim=in_dim,
                out_dim=hidden_dim,
                rank=6,
                lambda_r_init=0.1,
                lambda_q_init=0.1,
                use_r=True,
                use_q=True
            ))
            self.activations.append(nn.ReLU())
            in_dim = hidden_dim

        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = layer(x)
            x = act(x)
        x = self.head(x)
        return x

def train_chatbot():
    """Train chatbot QBNN."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Create vectorizer
    vectorizer = create_vectorizer()

    # Prepare data
    questions = [q for q, _, _ in CONVERSATIONS]
    responses = [c for _, c, _ in CONVERSATIONS]

    X = vectorizer.transform(questions).toarray().astype(np.float32)
    y = np.array(responses, dtype=np.int64)

    print(f"Input dimension: {X.shape[1]}")
    print(f"Number of classes: {len(RESPONSE_CATEGORIES)}")
    print(f"Training samples: {len(CONVERSATIONS)}")

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model
    model = ChatbotQBNN(input_dim=X.shape[1], hidden_dim=64, n_classes=12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    epochs = 150
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device))
        preds = logits.argmax(dim=1)
        acc = (preds == y_tensor.to(device)).float().mean().item()
    print(f"Training Accuracy: {acc:.2%}")

    # Save
    torch.save(model.state_dict(), "chatbot_qbnn.pkl")
    with open("chatbot_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Model and vectorizer saved")

    return model, vectorizer

if __name__ == "__main__":
    train_chatbot()
