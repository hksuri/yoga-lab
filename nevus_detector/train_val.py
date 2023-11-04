import os
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from config import parse_arguments
from utils import get_model, create_train_val_test_loaders, evaluate, plot_and_save_losses_accuracies, calculate_auroc, create_preds_plot
from gradcam import plot_gradcam

def train(args, csv_file, num_epochs, learning_rates, batch_sizes):

    best_val_loss = float("inf")
    best_model = None
    best_train_losses, best_train_accuracies, best_val_losses, best_val_accuracies = [], [], [], []
    test_image_paths, test_predicted_labels = [], []

    best_batch_size = 0
    best_learning_rate = 0.

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:

            # Create dataloaders
            train_loader, val_loader, test_loader, test_loader_gradcam = create_train_val_test_loaders(csv_file, batch_size)

            # Obtain model
            model = get_model(pretrained = args.pretrained_dir)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_model_found = 0

            train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

            for epoch in range(num_epochs):
                
                model.train()
                total_loss = 0
                correct_train = 0
                total_train = 0

                for images, labels, _ in train_loader:
                    
                    images, labels = images.to(device), labels.reshape((-1,1)).to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    targets = torch.stack((labels, 1 - labels), dim=1).squeeze(2)

                    # print(f'target size: {targets.size()}, output size: {outputs.size()}')
                    # print(f'labels: {labels.size()}, preds: {outputs.size()}')

                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    predicted = torch.argmax(torch.sigmoid(outputs).float(),dim=1)
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
                    best_model_found = 1

            if best_model_found:
                best_batch_size = batch_size
                best_learning_rate = learning_rate
                best_train_losses = train_losses.copy()
                best_train_accuracies = train_accuracies.copy()
                best_val_losses = val_losses.copy()
                best_val_accuracies = val_accuracies.copy()

    # Testing
    model.load_state_dict(best_model)

    _, _, test_gt_labels, test_predicted_labels, test_image_paths = evaluate(model, test_loader, criterion, device)
    auroc = calculate_auroc(test_gt_labels, test_predicted_labels)
    print(f"\nTest AUROC: {auroc:.4f} - Best Batch Size: {best_batch_size}, Best Learning Rate: {best_learning_rate}")

    return test_loader_gradcam, best_model, best_train_losses, best_train_accuracies, best_val_losses, best_val_accuracies, test_image_paths, test_predicted_labels

def main():

    # Create argument parsers
    args = parse_arguments()
    
    # Set the environment variable to store pretrained model
    os.environ['TORCH_HOME'] = args.main_dir + 'base_models'
    
    # Define directories
    # csv_file = main_dir + 'nevus_labels.csv'
    csv_file = args.main_dir + 'nevus_labels_mforge.csv'
    best_model_path = args.main_dir + 'nevus_detector_best_models'

    # Run experiment
    test_loader_gradcam, best_model_weights, train_losses, train_accuracies, val_losses, val_accuracies, test_image_paths, test_predicted_labels = train(args, csv_file, args.num_epochs, args.learning_rate, args.batch_size)

    # Get current date and time
    now = datetime.datetime.now()
    
    # Save the trained model weights to a file
    date_time_str = now.strftime("/best_model_weights_%m_%d_%y_%H_%M.pth")
    best_model_path += date_time_str
    torch.save(best_model_weights, best_model_path)
    
    # Plot and save train/val losses and accuracies
    plot_str = now.strftime('nevus_detector_plots/losses_accs_plot_%m_%d_%y_%H_%M.png')
    plot_save_directory = args.main_dir + plot_str
    plot_and_save_losses_accuracies(train_losses, val_losses, train_accuracies, val_accuracies, plot_save_directory)

    # Plot and save sample predictions
    preds_str = now.strftime('nevus_detector_preds/preds_%m_%d_%y_%H_%M.png')
    preds_directory = args.main_dir + preds_str
    create_preds_plot(test_image_paths, test_predicted_labels, csv_file, preds_directory)

    # Plot and save Grad-CAM output
    gradcam_str = now.strftime('nevus_detector_gradcam/gradcam_%m_%d_%y_%H_%M.png')
    gradcam_directory = args.main_dir + gradcam_str
    plot_gradcam(best_model_path, test_loader_gradcam, gradcam_directory)

if __name__ == "__main__":
    main()