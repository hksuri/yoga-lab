import torch

image_size = 244
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Distorted
num_classes_dr = 2 # Changed to binary problem
lr_init_dr = 1e-5
batch_size_dr = 64
num_epochs_dr = 5
weight_decay_dr = 1e-2 # Regularization parameter

# Good vs Bad
num_classes_gb = 2
lr_init_gb = 1e-7
batch_size_gb = 64
num_epochs_gb = 100
weight_decay_gb = 1e-2 # Regularization parameter