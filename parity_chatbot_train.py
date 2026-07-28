"""
Train a lightweight QBNN for the Parity task to use in the chatbot.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from apqb_qbnn_v2 import QBNNLayerV2
import pickle

def make_parity(n_samples, n_features=4, k=None):
    """Generate parity (XOR on first k features) task."""
    if k is None:
        k = n_features
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = (X[:, :k] > 0.5).sum(axis=1) % 2
    return torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))

class ParityQBNN(nn.Module):
    """Simple QBNN for Parity classification."""
    def __init__(self, n_features, hidden_dim=32, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        in_dim = n_features
        for i in range(n_layers):
            self.layers.append(QBNNLayerV2(
                in_dim=in_dim,
                out_dim=hidden_dim,
                rank=4,
                lambda_r_init=0.1,
                lambda_q_init=0.1,
                use_r=True,
                use_q=True
            ))
            self.activations.append(nn.ReLU())
            in_dim = hidden_dim

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = layer(x)
            x = act(x)
        x = self.head(x)
        return torch.sigmoid(x)

def train_parity_model(n_samples=2000, n_features=4, epochs=100, save_path="parity_qbnn.pkl"):
    """Train QBNN on parity task."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Generate data
    X, y = make_parity(n_samples, n_features, k=2)
    dataset = TensorDataset(X, y.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = ParityQBNN(n_features, hidden_dim=32, n_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # Train
    model.train()
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

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    # Evaluate
    X_test, y_test = make_parity(500, n_features, k=2)
    X_test, y_test = X_test.to(device), y_test.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(X_test) > 0.5
        acc = (preds.squeeze() == y_test).float().mean().item()
    print(f"Test Accuracy: {acc:.2%}")

    # Save
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

if __name__ == "__main__":
    train_parity_model()
