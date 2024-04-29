import torch
import torch.nn as nn
from torchvision import models

class CustomResNet(nn.Module):  # Inherit from nn.Module
    def __init__(self, model_name='resnet18', num_classes=2, checkpoint_path=None):
        super(CustomResNet, self).__init__()  # Initialize the superclass
        """
        Initialize the CustomResNet model.
        :param model_name: str, 'resnet18' or 'resnet50'
        :param num_classes: int, number of output classes (default 2 for binary classification)
        :param pretrained: bool, whether to load pretrained weights (default True)
        :param checkpoint_path: str, path to a checkpoint file to load (default None)
        """
        if model_name == 'resnet18':
            self.base_model = models.resnet18(weights='IMAGENET1K_V1')
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(weights='IMAGENET1K_V2')
        else:
            raise ValueError("Unsupported model type. Choose 'resnet18' or 'resnet50'.")

        # Store the number of features for the classifier and replace the classifier
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Set the original classifier to an identity layer

        # Create a new classifier for binary classification
        self.classifier = nn.Linear(num_features, num_classes)
        
        # If a checkpoint is provided, load the model weights from the checkpoint
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.
        :param checkpoint_path: str, path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: torch.Tensor, input tensor
        :return: tuple of (torch.Tensor, torch.Tensor), where the first tensor is the embedding and the second is the classification output
        """
        embeddings = self.base_model(x)
        output = self.classifier(embeddings)
        return embeddings, output