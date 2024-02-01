# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from random import shuffle

def build_dataset(is_train, mrn_list, args):
    
    transform = build_transform(is_train, args)
    dataset = CustomDataset(args.data_path, mrn_list, args.rf, transform)
    return dataset

# def build_dataset(is_train, args):
    
#     transform = build_transform(is_train, args)
#     root = os.path.join(args.data_path, is_train)
#     dataset = datasets.ImageFolder(root, transform=transform)

#     return dataset

def split_folders(directory_path):
    # Get the list of folder names in the specified directory
    folder_names = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]

    # Shuffle the list of folder names for random splitting
    shuffle(folder_names)

    # # Calculate the split indices for 80/20 split and 5 equal sets
    # split_index_80_20 = int(0.8 * len(folder_names))
    # split_indices_5_sets = [int(i * split_index_80_20 / 5) for i in range(1, 5)]
    split_indices_5_sets = [int(i * len(folder_names) / 5) for i in range(1, 5)]

    # # Divide the folder names into train_val and test lists
    # train_val = folder_names[:split_index_80_20]
    # test = folder_names[split_index_80_20:]

    # # Split the train_val into 5 equal sets
    # train_val_sets = [train_val[i:j] for i, j in zip([0] + split_indices_5_sets, split_indices_5_sets + [None])]
    train_val_sets = [folder_names[i:j] for i, j in zip([0] + split_indices_5_sets, split_indices_5_sets + [None])]

    # return train_val_sets, test

    return train_val_sets

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class CustomDataset(Dataset):

    def __init__(self, root_dir, mrn_list, rf, transform):
        self.root_dir = root_dir
        self.mrn_list = mrn_list
        self.rf_type = {'dia5':-6, 'intref':-5, 'orange':-4, 'va':-3, 'thick2':-2, 'srf':-1}
        self.rf = self.rf_type[rf]
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        for mrn in self.mrn_list:
            mrn_folder = os.path.join(self.root_dir, str(mrn))
            if os.path.exists(mrn_folder):
                for filename in os.listdir(mrn_folder):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        image_path = os.path.join(mrn_folder, filename)
                        label = int(filename.split('.')[0].split('_')[self.rf])
                        data.append((image_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)
                                    
        image = self.transform(image)
        return img_path, image, label
