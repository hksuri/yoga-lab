import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import pandas as pd
from PIL import Image
import os
import numpy as np
import json

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filepaths = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    self.filepaths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

def load_dataset(directory):
    transform = transforms.Compose([
        transforms.Resize((700, 700)),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.318, 0.154,0.073], std=[0.146,0.081,0.057]),
    ])
    dataset = CustomDataset(directory, transform)
    return dataset

def main():
    data_directory = '/research/labs/ophthalmology/iezzi/m294666/nevus_data_500_risk_factors_June2024_final'
    batch_size = 1
    dataset = load_dataset(data_directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load the DINO V2 model
    os.environ['TORCH_HOME'] = '/research/labs/ophthalmology/iezzi/m294666/base_models'

    model = torch.hub.load('/research/labs/ophthalmology/iezzi/m294666/base_models/hub/dinov2', 'dinov2_vitg14', source='local', pretrained=False)

    weights_path = '/research/labs/ophthalmology/iezzi/m294666/base_models/dinov2_vitg14_pretrain.pth'
    model.load_state_dict(torch.load(weights_path))  
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    embeddings = []
    image_names = []
    
    with torch.no_grad():
        for images, filenames in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            output = model(images)

            embeddings.append(output[0].cpu().numpy().tolist())
            image_names.extend(filenames)

    # Save embeddings and filenames in a JSON file
    save_path = '/research/labs/ophthalmology/iezzi/m294666/dino_embeddings.json'
    embeddings_dict = {
        'image_name': image_names,
        'embeddings': embeddings
    }
    with open(save_path, 'w') as f:
        json.dump(embeddings_dict, f)

    print('Embedding dimensions:', np.array(embeddings).shape)
    print(f'Embeddings saved to {save_path}')

if __name__ == '__main__':
    main()