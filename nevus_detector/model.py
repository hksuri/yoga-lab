import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet(nn.Module):
    def __init__(self, num_outputs=2, pretrained=None):
        super(CustomResNet, self).__init__()

        # Load the pre-trained ResNet model if a path is provided, or create a new one
        if pretrained is not None:
            self.model = models.resnet18()
            self.model.load_state_dict(torch.load(pretrained))
            print(f'\nWeights loaded from {pretrained}.\n')
        else:
            # self.model = models.resnet50(weights='IMAGENET1K_V1')
            self.model = models.resnet101(weights='IMAGENET1K_V2')
            print('\nInitialized model pretrained on ImageNet.\n')

        # Modify the final FC layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.model(x)


class CustomResNet18_GradCAM(nn.Module):
    def __init__(self, num_outputs=2, pretrained=None):
        super(CustomResNet18_GradCAM, self).__init__()

        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        # Load the pre-trained ResNet-18 model if a path is provided, or create a new one
        if pretrained is not None:
            self.model = models.resnet18()
            self.model.load_state_dict(torch.load(pretrained))
        else:
            self.model = models.resnet18(weights='IMAGENET1K_V1')

        # Modify the final FC layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_outputs)

        self.layerhook.append(self.model.layer4.register_forward_hook(self.forward_hook()))
        
        for p in self.model.parameters():
            p.requires_grad = True
        
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, x):
        out = self.model(x)
        return out, self.selected_out