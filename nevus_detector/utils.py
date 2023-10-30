import torch
from torch.utils.data import DataLoader, random_split
from dataloader import CustomDataset
from model import CustomResNet18
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
            
def get_model(num_outputs=1, pretrained=None, layer_names_to_freeze=[]):
    model = CustomResNet18(num_outputs)
    freeze_layers(model, layer_names_to_freeze)
#     print_trainable_layers(model)
    return model

def create_train_val_test_loaders(csv_file, batch_size, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, image_size=224):
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
    
    return train_loader, val_loader, test_loader

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
    total_loss = 0
    correct = 0
    total = 0
    gt_labels = []
    predicted_labels = []
    image_paths = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            gt_labels.extend(labels.tolist())
            images, labels = images.to(device), labels.reshape((-1,1)).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predicted_labels.extend(predicted.cpu().numpy())
            image_paths.extend(paths)

    loss = total_loss / len(dataloader)
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
    save_path = f"{save_dir}/losses_accuracies_plot.png"
    plt.savefig(save_path)
    print(f"Losses and accuracies plot saved at {save_path}")