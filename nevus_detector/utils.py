import os
import torch
import random
import pandas as pd
from torch.utils.data import DataLoader, random_split
from dataloader import CustomDataset
from model import CustomResNet, CustomResNet18_GradCAM
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def freeze_layers(model, layer_names_to_freeze):
    for name, param in model.named_parameters():
        if any(name.startswith(layer_name) for layer_name in layer_names_to_freeze):
            param.requires_grad = False

def print_trainable_layers(model):
    print("Trainable Layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            
def get_model(num_outputs=2, pretrained=None, layer_names_to_freeze=[]):
    model = CustomResNet(num_outputs)
    # freeze_layers(model, layer_names_to_freeze)
#     print_trainable_layers(model)
    return model

def get_model_gradcam(num_outputs=2, pretrained=None, layer_names_to_freeze=[]):
    model = CustomResNet18_GradCAM(num_outputs)
    freeze_layers(model, layer_names_to_freeze)
#     print_trainable_layers(model)
    return model

def create_train_val_test_loaders(csv_file, batch_size, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, image_size=224):
    """
    Create train, validation, and test data loaders based on provided ratios.

    Args:
        csv_file (str): Path to the csv file.
        batch_size (int): Number of samples in each batch.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.
        image_size (int): Size to which the image is resized.

    Returns:
        train_loader, val_loader, test_loader: DataLoader instances for train, validation, and test datasets.
    """
    dataset = CustomDataset(csv_file, image_size=image_size)
    
    # Calculate data split sizes
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    
    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoader instances for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader_gradcam = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Total images in train loader: {len(train_dataset)}")
    print(f"Total images in validation loader: {len(val_dataset)}")
    print(f"Total images in test loader: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, test_loader_gradcam

# def create_train_val_test_loaders(csv_file, batch_size, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, image_size=224):
#     """
#     Create train, validation, and test data loaders based on provided ratios.

#     Args:
#         csv_file (str): Path to the csv file.
#         batch_size (int): Number of samples in each batch.
#         train_ratio (float): Ratio of training data.
#         val_ratio (float): Ratio of validation data.
#         test_ratio (float): Ratio of test data.
#         image_size (int): Size to which the image is resized.

#     Returns:
#         train_loader, val_loader, test_loader: DataLoader instances for train, validation, and test datasets.
#     """
#     dataset = CustomDataset(csv_file, image_size=image_size)
    
#     # Create a dictionary to hold images for each patient
#     patient_images = {}
    
#     # Iterate through the dataset and group images by patient MRN
#     for idx, (image, label, image_path) in enumerate(dataset):
#         # Extract the patient MRN from the image path
#         patient_mrn = os.path.basename(os.path.dirname(image_path))
#         if patient_mrn not in patient_images:
#             patient_images[patient_mrn] = []
#         patient_images[patient_mrn].append((image, label, image_path))
    
#     # Shuffle the patient MRNs to distribute them randomly
#     patient_mrns = list(patient_images.keys())
#     random.shuffle(patient_mrns)
    
#     # Calculate data split sizes based on patient count
#     total_patients = len(patient_mrns)
#     train_size = int(train_ratio * total_patients)
#     val_size = int(val_ratio * total_patients)
#     test_size = total_patients - train_size - val_size
    
#     # Split the patient MRNs into train, validation, and test sets
#     train_mrns = patient_mrns[:train_size]
#     val_mrns = patient_mrns[train_size:train_size + val_size]
#     test_mrns = patient_mrns[train_size + val_size:]
    
#     # Create DataLoader instances for train, validation, and test datasets
#     train_dataset = [(image, label, image_path) for mrn in train_mrns for image, label, image_path in patient_images[mrn]]
#     val_dataset = [(image, label, image_path) for mrn in val_mrns for image, label, image_path in patient_images[mrn]]
#     test_dataset = [(image, label, image_path) for mrn in test_mrns for image, label, image_path in patient_images[mrn]]
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     test_loader_gradcam = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
#     # Calculate the total size of images in each dataloader
#     total_train_images = len(train_dataset)
#     total_val_images = len(val_dataset)
#     total_test_images = len(test_dataset)
    
#     print(f"Total images in train loader: {total_train_images}")
#     print(f"Total images in validation loader: {total_val_images}")
#     print(f"Total images in test loader: {total_test_images}")
    
#     return train_loader, val_loader, test_loader, test_loader_gradcam


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate a model using a DataLoader.

    Args:
        model: The binary classification model.
        dataloader: DataLoader for validation or test data.
        criterion: Loss function (e.g., BCEWithLogitsLoss).
        device: The device on which to perform the evaluation (e.g., "cuda" or "cpu").

    Returns:
        loss: The average loss over the data.
        accuracy: The accuracy of the model on the data.
        gt_labels: The ground truth labels of the data.
        predicted_labels: List of predicted labels (0 or 1) for the data.
        image_paths: List of file paths for the data.
    """
    model.eval()
    total_loss = 0.
    correct = 0
    total = 0
    gt_labels = []
    predicted_labels = []
    image_paths = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            gt_labels.extend(labels.tolist())
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)
            outputs = model(images)
            # targets = torch.stack((labels, 1 - labels), dim=1).squeeze(2)

            # loss = criterion(outputs, targets)
            loss = criterion(outputs,labels)
            total_loss += loss.item()

            # predicted = torch.argmax((torch.sigmoid(outputs) > 0.5).float(), dim=1)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predicted_labels.extend(predicted.cpu().numpy())
            image_paths.extend(paths)

            # print(f'predicted labels eval: {predicted}')

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy, gt_labels, predicted_labels, image_paths

def calculate_auroc(labels, predicted_labels):
    """
    Calculate the Area Under the Receiver Operating Characteristic (AUROC) score.

    Args:
        labels: Ground truth labels (0 or 1).
        predicted_labels: Predicted labels (0 or 1) from the model.

    Returns:
        auroc: AUROC score.
    """
    auroc = roc_auc_score(labels, predicted_labels)
    return auroc

def plot_and_save_losses_accuracies(train_losses, val_losses, train_accuracies, val_accuracies, save_dir):
    """
    Plot train and validation losses and accuracies and save the plots.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_accuracies (list): List of training accuracies.
        val_accuracies (list): List of validation accuracies.
        save_dir (str): Directory to save the plots.
    """
    plt.figure(figsize=(12, 5))
    
    x = list(range(1, len(train_losses) + 1))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(x,train_losses, label='Training Loss')
    plt.plot(x,val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(x,train_accuracies, label='Training Accuracy')
    plt.plot(x,val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracies')

    # Save the plots
    # save_path = f"{save_dir}/losses_accuracies_plot.png"
    plt.savefig(save_dir)
    print(f"Losses and accuracies plot saved at {save_dir}")

def create_preds_plot(test_image_paths, test_predicted_labels, csv_file_path, save_dir):
    """
    Plot the predictions on sample test images and save the plot.

    Args:
        test_image_paths (list): List of paths of test images.
        test_predicted_labels (list): List of predicted labels.
        csv_file_path (str): Directory to csv file containing ground truth labels.
        save_dir (str): Directory to save the plot.
    """
    # Read the CSV file and create a DataFrame
    df = pd.read_csv(csv_file_path)

    # Obtain the ground truth labels associated with test_image_paths
    ground_truth_labels = []
    for image_path in test_image_paths:
        row = df[df['Image Path'] == image_path]
        if not row.empty:
            ground_truth_labels.append(row['Label'].values[0])

    # Randomly choose 16 images
    # random.seed(42)  # Set a seed for reproducibility
    selected_indices = random.sample(range(len(test_image_paths)), 16)

    # Create a 2x4 subplot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    # Loop through selected images and plot them
    for i, index in enumerate(selected_indices):
        image_path = test_image_paths[index]
        predicted_label = test_predicted_labels[index]
        ground_truth_label = ground_truth_labels[index]
        ax = axes[i // 4, i % 4]
        ax.axis('off')
        ax.set_title(f'Prediction: {int(predicted_label)}', color='green' if predicted_label == ground_truth_label else 'red')
        img = plt.imread(image_path)
        ax.imshow(img)

    # Save the subplot to the specified directory
    fig.tight_layout()
    plt.savefig(save_dir)
    plt.close()
    print(f'\nPreds saved to {save_dir}')