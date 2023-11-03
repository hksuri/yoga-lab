import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet18(nn.Module):
    def __init__(self, num_outputs=2, pretrained=None):
        super(CustomResNet18, self).__init__()

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

        # Modify the final FC layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_outputs)
        
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