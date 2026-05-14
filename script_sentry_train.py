import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from script_sentry_model import ScriptSentry


def train():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = ScriptSentry()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    print("Training Script-Sentry Forensic Engine (20 Epochs)...")

    for epoch in range(20):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/20 - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "scriptsentry_v1.pth")
    print("Model saved. Now ready for Forgery Detection.")

    plt.plot(loss_history)
    plt.title("Forensic Training Convergence")
    plt.show()


if __name__ == "__main__":
    train()