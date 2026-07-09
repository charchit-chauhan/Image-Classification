import torch.nn as nn
from torchvision import models


def get_model(num_classes=5):
    """
    Loads a pretrained ResNet-18 model and replaces
    the final classification layer.
    """

    # Load pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model