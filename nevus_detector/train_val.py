import os
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from config import parse_arguments
from utils import get_model, create_train_val_test_loaders, evaluate, plot_and_save_losses_accuracies, calculate_auroc

def train(model, train_loader, val_loader, test_loader, num_epochs, learning_rate, batch_size):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model = None
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    test_image_paths, test_predicted_labels = [], []

    for epoch in range(num_epochs):
        
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            
            images, labels = images.to(device), labels.reshape((-1,1)).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # print(f'labels: {labels.size()}, preds: {outputs.size()}')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Validation
        val_loss, val_accuracy, _, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    # Testing
    model.load_state_dict(best_model)

    test_loss, _, test_gt_labels, test_predicted_labels, test_image_paths = evaluate(model, test_loader, criterion, device)
    auroc = calculate_auroc(test_gt_labels, test_predicted_labels)
    print(f"\nTest AUROC: {auroc:.4f} - Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    return best_model, train_losses, train_accuracies, val_losses, val_accuracies, test_image_paths, test_predicted_labels

def main():

    # Create argument parsers
    args = parse_arguments()
    
    # Set the environment variable to store pretrained model
    os.environ['TORCH_HOME'] = args.main_dir + 'base_models'
    
    # Define directories
    # csv_file = main_dir + 'nevus_labels.csv'
    csv_file = args.main_dir + 'nevus_labels_mforge.csv'
    best_model_path = args.main_dir + 'nevus_detector_best_models'

    # Create dataloaders
    train_loader, val_loader, test_loader = create_train_val_test_loaders(csv_file, args.batch_size)

    # Obtain model
    model = get_model(pretrained = args.pretrained_dir)
    best_model_weights, train_losses, train_accuracies, val_losses, val_accuracies, test_image_paths, test_predicted_labels = train(model, train_loader, val_loader, test_loader, args.num_epochs, args.learning_rate, args.batch_size)

    # Save the trained model weights to a file
    now = datetime.datetime.now()
    date_time_str = now.strftime("/best_model_weights_%m_%d_%y_%H_%M.pth")
    best_model_path += date_time_str
    torch.save(best_model_weights, best_model_path)
    
    # Plot and save train/val losses and accuracies
    plot_str = now.strftime('nevus_detector_plots/losses_accs_plot_%m_%d_%y_%H_%M.png')
    plot_save_directory = args.main_dir + plot_str
    plot_and_save_losses_accuracies(train_losses, val_losses, train_accuracies, val_accuracies, plot_save_directory)

if __name__ == "__main__":
    main()