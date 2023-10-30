import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet18(nn.Module):
    def __init__(self, num_outputs=2, pretrained=None):
        super(CustomResNet18, self).__init__()

        # Load the pre-trained ResNet-18 model if a path is provided, or create a new one
        if pretrained is not None:
            self.model = models.resnet18()
            self.model.load_state_dict(torch.load(pretrained))
        else:
            self.model = models.resnet18(weights='IMAGENET1K_V1')

        # Modify the final FC layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.model(x)