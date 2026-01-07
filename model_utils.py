import os
from typing import Tuple, Dict

from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_datasets(dataset_path: str) -> Tuple[TensorDataset, Dict[str, int], Dict[int, str]]:
    """Load images from subfolders as classes and return dataset plus label maps."""
    transform = get_transforms()
    images = []
    labels = []

    folders = sorted(os.listdir(dataset_path))
    char_to_idx = {folder: i for i, folder in enumerate(folders)}
    idx_to_char = {i: folder for i, folder in enumerate(folders)}

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if not os.path.isfile(img_path):
                continue
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img = Image.open(img_path)
            img = transform(img)
            images.append(img)
            labels.append(char_to_idx[folder])

    if not images:
        raise ValueError("No images found in dataset path")

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(images_tensor, labels_tensor)
    return dataset, char_to_idx, idx_to_char
