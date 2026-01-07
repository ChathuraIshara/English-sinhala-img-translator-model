import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

dataset_path = "dataset"  # your folder path
images = []
labels = []

folders = sorted(os.listdir(dataset_path))
char_to_idx = {folder:i for i, folder in enumerate(folders)}
idx_to_char = {i:folder for i, folder in enumerate(folders)}

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)

    # skip if not a directory
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        # skip directories like .ipynb_checkpoints
        if not os.path.isfile(img_path):
            continue

        # optional: allow only image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = Image.open(img_path)
        img = transform(img)
        images.append(img)
        labels.append(char_to_idx[folder])

for folder in folders:
    print(folder, "→", os.listdir(os.path.join(dataset_path, folder)))

# Convert lists to tensors
images = torch.stack(images)      # shape: [N, 1, 32, 32]
labels = torch.tensor(labels)     # shape: [N]

# Create dataset
dataset = TensorDataset(images, labels)

# Train / test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=len(folders)).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for imgs, lbls in train_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)

print(f"Test Accuracy: {correct/total*100:.2f}%")

def predict_image(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()

    letter = idx_to_char[pred]
    plt.imshow(img.cpu().squeeze(), cmap='gray')
    plt.title(f"Prediction: {letter}")
    plt.axis('off')
    plt.show()
    print("Predicted Letter:", letter)

predict_image("new.jpg")



