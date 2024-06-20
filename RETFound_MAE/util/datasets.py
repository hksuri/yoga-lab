# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import random
from random import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import torch

def extract_class_from_image_name(image_name, rf):
    rf_type = {'dia5':-6, 'intref':-5, 'orange':-4, 'va':-3, 'thick2':-2, 'srf':-1}
    rf = rf_type[rf]
    label = int(image_name.split('.')[0].split('_')[rf])
    return label

def determine_mrn_classes(main_directory, rf):

    if rf == 'nevusNoNevus':
        return determine_mrn_classes_nevus_nonevus(main_directory)
    else:
        """Determine the class of each MRN based on the first image in its directory."""
        mrn_classes = {}
        for mrn_folder in os.listdir(main_directory):
            mrn_path = os.path.join(main_directory, mrn_folder)
            if os.path.isdir(mrn_path):
                images = sorted(os.listdir(mrn_path))
                if images:
                    reference_image = images[0]
                    class_name = extract_class_from_image_name(reference_image, rf)
                    mrn_classes[mrn_folder] = class_name
        return mrn_classes

def create_nested_stratified_folds(mrn_classes, outer_splits=5, inner_split_ratio=0.875):
    """Create nested stratified folds with a structure for train, validation, and test sets."""

    mrns = list(mrn_classes.keys())
    classes = list(mrn_classes.values())

    if outer_splits == 1:
        train_mrn, test_mrn = train_test_split(mrns, test_size=0.2, stratify=classes)
        return [train_mrn, test_mrn]
    
    else:
        mrns = np.array(mrns)
        classes = np.array(classes)
        outer_skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
        
        folds = []
        for trainval_index, test_index in outer_skf.split(mrns, classes):
            trainval_mrns, test_mrns = mrns[trainval_index], mrns[test_index]
            trainval_classes = classes[trainval_index]

            inner_skf = StratifiedKFold(n_splits=int(1/(1-inner_split_ratio)), shuffle=True, random_state=42)
            train_index, val_index = next(inner_skf.split(trainval_mrns, trainval_classes))

            train_mrns, val_mrns = trainval_mrns[train_index], trainval_mrns[val_index]
            
            folds.append([train_mrns.tolist(), val_mrns.tolist(), test_mrns.tolist()])
            
        return folds

def calculate_weights(mrn_classes):
    # Counting the occurrences of each class
    class_counts = Counter(mrn_classes.values())

    # Calculate the total number of entries
    total_entries = sum(class_counts.values())

    # Determining weights inversely proportional to the frequency of each class
    weights = {
        class_label: total_entries / (class_count * len(class_counts))
        for class_label, class_count in class_counts.items()
    }

    # Converting weights to a tensor
    weight_tensor = torch.tensor([weights[i] for i in sorted(weights.keys())], dtype=torch.float)
    return weight_tensor

# def build_dataset(is_train, mrn_list, args):
    
#     transform = build_transform(is_train, args)
#     dataset = CustomDataset(args.data_path, mrn_list, args.rf, transform)
#     return dataset

def build_dataset(is_train, mrn_list, args):
    transform = build_transform(is_train, args)
    if args.rf == 'nevusNoNevus':
        dataset = Nevus_NoNevus_Dataset(args.data_path, mrn_list, transform, is_train)
    else:
        dataset = RFDataset(args.data_path, mrn_list, args.rf, transform, is_train)
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
    # mean = IMAGENET_DEFAULT_MEAN
    # std = IMAGENET_DEFAULT_STD
    mean = [0.318, 0.154,0.073]
    std = [0.146,0.081,0.057]
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

    ###################### ZOOM ########################

    if args.zoom == 'in':

        # Zoom In

        # Calculate the resized size
        resize_size = int(1 / (1 - args.zoom_level) * args.input_size)
        # Resize the image to zoom in
        t.append(transforms.Resize(resize_size,interpolation=transforms.InterpolationMode.BICUBIC)) 
        # Center crop to maintain the original resolution            
        t.append(transforms.CenterCrop(args.input_size)) 
        # Resize back to original size (for uniformity)                                                      
        t.append(transforms.Resize(args.input_size,interpolation=transforms.InterpolationMode.BICUBIC)) 

    elif args.zoom == 'out':

        # Zoom Out

        # Calculate the resized size
        resize_size = int((1 - args.zoom_level) * args.input_size)
        padding = (args.input_size - resize_size) // 2
        # Resize the image to zoom in
        t.append(transforms.Resize(resize_size,interpolation=transforms.InterpolationMode.BICUBIC)) 
        # Add padding
        t.append(transforms.Pad(padding, fill=0, padding_mode='constant')) 
        # Center crop to maintain the original resolution            
        t.append(transforms.CenterCrop(args.input_size)) 
        # Resize back to original size (for uniformity)                                                      
        t.append(transforms.Resize(args.input_size,interpolation=transforms.InterpolationMode.BICUBIC))                                                           
    
    else:
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
        )
        t.append(transforms.CenterCrop(args.input_size))

    ####################################################

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))

    # # List of additional transforms
    # additional_transforms = [
    #     transforms.RandomHorizontalFlip(p=1.0),
    #     transforms.RandomVerticalFlip(p=1.0),
    #     transforms.RandomRotation(degrees=90),
    #     transforms.RandomAffine(degrees=0, shear=20),
    #     transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
    #     transforms.ColorJitter(contrast=0.5),
    # ]
    
    # # Compute the probability for each random transform
    # N = len(additional_transforms)
    # prob = 1.0 / N
    
    # # Append each transform with the computed probability
    # for transform in additional_transforms:
    #     t.append(transforms.RandomApply([transform], p=prob))

    return transforms.Compose(t)

# class CustomDataset(Dataset):

#     def __init__(self, root_dir, mrn_list, rf, transform):
#         self.root_dir = root_dir
#         self.mrn_list = mrn_list
#         self.rf_type = {'dia5':-6, 'intref':-5, 'orange':-4, 'va':-3, 'thick2':-2, 'srf':-1}
#         self.rf = self.rf_type[rf]
#         self.transform = transform
#         self.data = self.load_data()

#     def load_data(self):
#         data = []
#         for mrn in self.mrn_list:
#             mrn_folder = os.path.join(self.root_dir, str(mrn))
#             if os.path.exists(mrn_folder):
#                 for filename in os.listdir(mrn_folder):
#                     if filename.endswith(".jpg") or filename.endswith(".png"):
#                         image_path = os.path.join(mrn_folder, filename)
#                         label = int(filename.split('.')[0].split('_')[self.rf])
#                         data.append((image_path, label))
#         return data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path, label = self.data[idx]
#         image = Image.open(img_path)
                                    
#         image = self.transform(image)
#         return img_path, image, label

class RFDataset(Dataset):

    def __init__(self, root_dir, mrn_list, rf, transform, is_train):
        self.root_dir = root_dir
        self.mrn_list = mrn_list
        self.rf_type = {'dia5':-6, 'intref':-5, 'orange':-4, 'va':-3, 'thick2':-2, 'srf':-1}
        self.rf = self.rf_type[rf]
        self.transform = transform
        self.data = self.load_data()
        if is_train=='train':
            self.data = self.oversample_data()
        class_counts = {0: 0, 1: 0}
        for _, label in self.data:
            class_counts[label] += 1
        print(f"Class counts: {class_counts}")

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

    def oversample_data(self):
        class_counts = {0: 0, 1: 0}
        for _, label in self.data:
            class_counts[label] += 1

        # Find the majority class
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = 1 - majority_class

        # Calculate oversampling ratio
        oversample_ratio = class_counts[majority_class] // class_counts[minority_class]

        oversampled_data = []
        for item in self.data:
            if item[1] == minority_class:
                # Oversample minority class items
                oversampled_data.extend([item] * oversample_ratio)
            else:
                oversampled_data.append(item)

        # Count and print number of items in each class
        oversampled_class_counts = {0: 0, 1: 0}
        for _, label in oversampled_data:
            oversampled_class_counts[label] += 1
        print(f"Class counts before oversampling: {class_counts}")
        print(f"Class counts after oversampling: {oversampled_class_counts}")

        return oversampled_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)                                    
        image = self.transform(image)
        return img_path, image, label
    
################################## Nevus Detector #############################################

def determine_mrn_classes_nevus_nonevus(root_dir):
    class_dirs = {
        'nevus_data_500_risk_factors_June2024_final': 1,
        'Dalvin_nevus_std_Normal_COLOR_final_processed': 0
    }
    mrn_class_count = defaultdict(Counter)
    for dir_name, class_label in class_dirs.items():
        class_dir = os.path.join(root_dir, dir_name)
        for mrn in os.listdir(class_dir):
            mrn_dir = os.path.join(class_dir, mrn)
            if os.path.isdir(mrn_dir):
                # Count each image for the MRN under this class
                mrn_class_count[mrn][class_label] += len(os.listdir(mrn_dir))
    
    # Determine predominant class for each MRN
    mrn_to_class = {}
    for mrn, counts in mrn_class_count.items():
        # Select the class with the maximum count
        predominant_class = counts.most_common(1)[0][0]
        mrn_to_class[mrn] = predominant_class
    
    return mrn_to_class
    
class Nevus_NoNevus_Dataset(Dataset):
    def __init__(self, root_dir, mrns, transform, is_train):
        self.root_dir = root_dir
        self.mrns = mrns
        self.transform = transform
        self.data = []
        self._load_data()

        # if is_train=='train':
        #     self.data = self.oversample_data()

        # Print class counts in data
        class_counts = Counter([label for _, label in self.data])
        print(f"Class counts: {class_counts}")

    def _load_data(self):
        # Load data from the two directories
        class_dirs = {
            'nevus_data_500_risk_factors_June2024_final': 1,
            'Dalvin_nevus_std_Normal_COLOR_final_processed': 0
        }
        for dir_name, label in class_dirs.items():
            class_dir = os.path.join(self.root_dir, dir_name)
            for mrn in self.mrns:
                mrn_dir = os.path.join(class_dir, mrn)
                if os.path.isdir(mrn_dir):
                    for img_name in os.listdir(mrn_dir):
                        img_path = os.path.join(mrn_dir, img_name)
                        self.data.append((img_path, label))

    def oversample_data(self):
        class_counts = {0: 0, 1: 0}
        for _, label in self.data:
            class_counts[label] += 1

        # Find the majority class
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = 1 - majority_class

        # Calculate oversampling ratio
        oversample_ratio = class_counts[majority_class] // class_counts[minority_class]

        oversampled_data = []
        for item in self.data:
            if item[1] == minority_class:
                # Oversample minority class items
                oversampled_data.extend([item] * oversample_ratio)
            else:
                oversampled_data.append(item)

        # Count and print number of items in each class
        oversampled_class_counts = {0: 0, 1: 0}
        for _, label in oversampled_data:
            oversampled_class_counts[label] += 1
        print(f"Class counts before oversampling: {class_counts}")
        print(f"Class counts after oversampling: {oversampled_class_counts}")

        return oversampled_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return img_path, image, label
