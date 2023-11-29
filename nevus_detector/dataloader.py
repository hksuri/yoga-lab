import torch
import cv2
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def crop_image(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def transform(image, image_size=224, sigmaX=8):
    """
    Preprocess and transform the input image.
    
    Args:
        image (PIL Image): The input image.
        image_size (int): Size to which the image is resized.
        
    Returns:
        torch.Tensor: Transformed image tensor.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Circular Crop
    image = crop_image(image)

    # Resize
    image = cv2.resize(image, (image_size, image_size))
    
    # Brightness / Contrast Enhancement
    # brightness_factor = 2.0
    # contrast_factor = 1.5 
    # image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)
    
    # CLAHE
    # r, g, b = cv2.split(image)
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(6, 6))
    # enhanced_g = clahe.apply(g)
    # image = cv2.merge((r, enhanced_g, b))

    # Unharp Masking
    # image=cv2.addWeighted ( image, 4, cv2.GaussianBlur( image , (0,0) , sigmaX) , -4, 128)

    # Convert the NumPy array back to a PIL image
    image = Image.fromarray(image)

    # Augmentations
    preprocess = transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.RandomRotation((-360, 360)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
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
        image = cv2.imread(img_path)  # Load image using the file path
        image = transform(image, self.image_size)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label, img_path