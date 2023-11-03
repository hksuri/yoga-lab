import os
import torch
import torchvision
import datetime
from torchvision import models, transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

from config import parse_arguments
from utils import get_model, create_train_val_test_loaders

# Create a function to generate Grad-CAM
def generate_gradcam(model, img, target_layer, threshold):
    # Forward pass
    output = model(img)
    model.zero_grad()
    
    # Calculate the gradient of the output with respect to the target layer
    output.backward(gradient=torch.ones_like(output))

    # Get the gradients at the target layer
    grads = target_layer.weight.grad

    # Global average pooling
    pooled_grads = F.adaptive_avg_pool2d(grads, (1, 1))
    
    # Get the activation map from the target layer
    activation_map = target_layer(img)
    
    for i in range(activation_map.size(1)):
        activation_map[:, i, :, :] *= pooled_grads[:, i, :, :]

    heatmap = activation_map.mean(dim=1, keepdim=True)
    
    heatmap = F.relu(heatmap)

    # Apply thresholding
    heatmap = torch.where(heatmap > threshold, heatmap, torch.zeros_like(heatmap))

    return heatmap

# Define a function to plot images with Grad-CAM
def plot_gradcam(images, heatmap, predictions, labels, file_name):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i in range(4):
        for j in range(4):
            idx = random.randint(0, len(images) - 1)
            image = images[idx]
            ax = axes[i, j]
            ax.axis('off')
            ax.imshow(image.transpose(1, 2, 0))
            ax.imshow(heatmap[idx].squeeze().cpu().data.numpy(), cmap='viridis', alpha=0.5, interpolation='bilinear')

            # Get the prediction and label for this image
            prediction = predictions[idx]
            label = labels[idx]

            # Set the title color based on matching prediction and label
            title_color = 'green' if prediction == label else 'red'
            ax.set_title(f'Prediction: {prediction}, Label: {label}', color=title_color)

    fig.tight_layout()
    plt.savefig(file_name)
    plt.close()

def main():

    args = parse_arguments()

    # Set the environment variable to store pretrained model
    os.environ['TORCH_HOME'] = args.main_dir + 'base_models'

    # Create dataloaders
    csv_file = args.main_dir + 'nevus_labels_mforge.csv'
    _, _, test_loader = create_train_val_test_loaders(csv_file, 1)

    # Load the ResNet-18 model
    model = get_model(pretrained = '/research/labs/ophthalmology/iezzi/m294666/nevus_detector_best_models/best_model_weights_11_02_23_20_11.pth')
    model.eval()

    # Generate Grad-CAM for a batch of test images
    test_images, test_labels = [], []
    for i, (image, label, _) in enumerate(test_loader):
        if i == 16:
            break
        test_images.append(image)
        test_labels.append(label)

    # Convert the images and labels to tensors
    test_images = torch.stack(test_images, dim=0)
    test_images = test_images.squeeze(1)
    test_labels = torch.cat(test_labels).int()

    print('test images shape:', test_images.shape)

    # Specify the target layer (modify this according to your model's architecture)
    target_layer =  model.model.layer4[1].conv2  # You may need to adjust the layer according to your model's architecture

    # Define the threshold for heatmap visualization
    threshold = 0.5  # Adjust this threshold as needed

    # Generate Grad-CAM for the images with thresholding
    heatmap = generate_gradcam(model, test_images, target_layer, threshold)

    # Make predictions for the test images
    with torch.no_grad():
        predictions = model(test_images).argmax(1).cpu().numpy()

    # Plot the images with Grad-CAM and colored titles
    now = datetime.datetime.now()
    gradcam_str = now.strftime('nevus_detector_preds/gradcam_%m_%d_%y_%H_%M.png')
    gradcam_path = args.main_dir + gradcam_str
    plot_gradcam(test_images, heatmap, predictions, test_labels, gradcam_path)

if __name__ == "__main__":
    main()