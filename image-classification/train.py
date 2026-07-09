import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import COCODataset
from model import get_model

# -----------------------
# Hyperparameters
# -----------------------
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -----------------------
# Dataset
# -----------------------
train_dataset = COCODataset(
    image_dir="data/train2017",
    label_file="train_labels.json"
)

val_dataset = COCODataset(
    image_dir="data/val2017",
    label_file="val_labels.json"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------
# Model
# -----------------------
model = get_model()
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_accuracy = 0

loss_history = []
accuracy_history = []

# -----------------------
# Training
# -----------------------
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    accuracy_history.append(accuracy)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Validation Accuracy: {accuracy:.2f}%"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(model.state_dict(), "best_model.pth")

        print("Best model saved!")

# -----------------------
# Save Graphs
# -----------------------

plt.figure(figsize=(6,4))
plt.plot(loss_history, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss.png")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(accuracy_history, marker='o')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig("accuracy.png")
plt.close()

print("\nTraining Complete!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

print("Saved:")
print("best_model.pth")
print("loss.png")
print("accuracy.png")