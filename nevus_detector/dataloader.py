import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

def transform(image, image_size=224):
    """
    Preprocess and transform the input image.
    
    Args:
        image (PIL Image): The input image.
        image_size (int): Size to which the image is resized.
        
    Returns:
        torch.Tensor: Transformed image tensor.
    """
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    return preprocess(image)

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_size=224):
        """
        Custom dataset for loading images and labels from an Excel file.

        Args:
            csv_file (str): Path to the csv file containing image paths and labels.
            image_size (int): Size to which the image is resized.
        """
        self.data = pd.read_csv(csv_file)
        self.image_size = image_size
        self.file_paths = self.data.iloc[:, 0].values
        self.labels = self.data.iloc[:, 1].values
        self.targets = self.labels.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data.iloc[idx]
        image = Image.open(img_path)  # Load image using the file path
        image = transform(image, self.image_size)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label, img_path