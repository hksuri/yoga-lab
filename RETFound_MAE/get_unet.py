import torch
import torch.nn as nn
from unet import UNet

class UnetEncoderHead(nn.Module):
    def __init__(self, checkpoint_path, args, n_channels, n_classes=3, output_classes=2, bilinear=False):
        super(UnetEncoderHead, self).__init__()
        # Initialize the full UNet
        self.unet = UNet(n_channels, n_classes, bilinear)

        # Load the checkpoint
        state_dict = torch.load(checkpoint_path, map_location=args.device)
        self.load_partial_state_dict(state_dict)

        # Determine the output dimension of the last encoder layer (needs manual setup based on UNet)
        output_dim = 1024 * 14 * 14  # This should be set based on your model configuration

        self.head = nn.Sequential(
            nn.Flatten(),  # Flatten the input
            nn.Linear(output_dim, 1024),  # First linear layer from flattened input to 512 units
            nn.GELU(),  # GELU activation
            nn.Dropout(p=0.3),  # Dropout layer with p=0.3
            nn.Linear(1024, 64),  # Second linear layer from 512 units to 64 units
            nn.GELU(),  # GELU activation
            nn.Dropout(p=0.3),  # Dropout layer with p=0.3
            nn.Linear(64, output_classes),  # Second linear layer from 64 units to output classes (2 in your case)
        )

    def forward(self, x):
        # Pass input through the encoder only
        # x1 = self.unet.inc(x)
        # x2 = self.unet.down1(x1)
        # x3 = self.unet.down2(x2)
        # x4 = self.unet.down3(x3)
        # x5 = self.unet.down4(x4)
        x5 = self.unet(x)

        # Pass through the custom head
        out = self.head(x5)

        return x5, out
    
    def load_partial_state_dict(self, state_dict):
        """Loads state dict selectively for the encoder parts, ignoring decoder."""
        model_state = self.unet.state_dict()
        # Use only keys that are part of the encoder
        encoder_keys = ['inc', 'down1', 'down2', 'down3', 'down4']
        filtered_state_dict = {k: v for k, v in state_dict.items() if any(k.startswith(key) for key in encoder_keys)}
        model_state.update(filtered_state_dict)
        self.unet.load_state_dict(model_state, strict=False)


# Example usage:
# model = UnetEncoderHead('path_to_checkpoint.pth', n_channels=3, n_classes=1, output_classes=10)
