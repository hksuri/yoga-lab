import os
import csv
import cv2
import time
import glob
import random
import shutil
import numpy as np
import pandas as pd
from skimage import io
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import torch
import torchvision
import torchmetrics
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import *
from config import *

# Class to load distorted dataset (stage 1)
class EyeData(Dataset):
    
    # initialize
    def __init__(self, data, root_dir, transform = None, do_random_crop = True, itype = '.png'):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.do_random_crop = do_random_crop
        self.itype = itype
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root_dir))):
            if not os.path.isdir(os.path.join(root_dir, name)):
                continue
            # len(..) returns from 0 - n based on saved labels so far
            self.name2label[name] = len(self.name2label.keys())
        print('labels: ',self.name2label)  
        self.images, self.labels = self.load_csv('/mnt/ssd_4tb_0/huzaifa/retina_kaggle/distorted_v2.csv')
        
    # length
    def __len__(self):
        # return len(self.data)
        # print(len(self.images))
        return len(self.images)
    
    # get items    
    # def __getitem__(self, idx):
    # img_name = os.path.join(self.directory, self.data.loc[idx, 'id_code'] + self.itype)
    # image    = prepare_image(img_name, do_random_crop = self.do_random_crop)
    # image    = self.transform(image)
    # label    = torch.tensor(self.data.loc[idx, 'diagnosis'])
    # return {'image': image, 'label': label}
    
    def __getitem__(self,index):
        img, label = self.images[index], self.labels[index]
        image = prepare_image(img, self.root_dir, do_random_crop = self.do_random_crop)
        label = torch.tensor(label)
        image = self.transform(image)
        return {'image': image, 'label': label}
    
    def load_csv(self,filename):
        # if no .csv, create; else load
        if not os.path.exists(filename):
            print('loading...')
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root_dir, name, '*.png'))
                images += glob.glob(os.path.join(self.root_dir, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root_dir, name, '*.jpeg'))
                # images += glob.glob(os.path.join(self.root_dir, name))
            print('images: ',len(images))
            random.shuffle(images)  
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    print('img: ',img)
                    name = img.split(os.sep)[-2]
                    # print('name: ',name)
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print(f"written into csv file: {filename}")       
        # read from csv file
        images, labels = [], []
        with open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)       
        assert len(images) == len(labels)
        return images, labels

# Class to load good vs bad data (stage 2)
class GoodBadDataset(Dataset):
    
    def __init__(self, root_dir, csv_dir, transform=None, do_random_crop = True, itype = '.png'):
        self.root_dir = root_dir
        self.transform = transform
        self.do_random_crop = do_random_crop
        self.itype = itype
        self.name2label = {}
        self.csv_dir = csv_dir
        
        for name in sorted(os.listdir(os.path.join(root_dir))):
            if not os.path.isdir(os.path.join(root_dir, name)):
                continue
            # len(..) returns from 0 - n based on saved labels so far
            temp = len(self.name2label.keys())
            if temp == 0 : self.name2label[name] = 1
            else: self.name2label[name] = 0
        print(self.name2label)
        # image, label, helper function load_csv()
        self.images, self.labels = self.load_csv(csv_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img, label = self.images[index], self.labels[index]
        
        # image = io.imread(img)
        label = torch.tensor(label)
        img, label = self.images[index], self.labels[index]
        image = prepare_image_goodbad(img, self.root_dir, do_random_crop = self.do_random_crop)
        image = image.float()
        label = torch.tensor(label)
        return {'image': image, 'label': label}
    
    def load_csv(self,filename):
        # if no .csv, create; else load
        if not os.path.exists(filename):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root_dir, name, '*.png'))
                images += glob.glob(os.path.join(self.root_dir, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root_dir, name, '*.jpeg'))

            print(len(images), images)
            random.shuffle(images)
            
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])
                print(f"written into csv file: {filename}")
                
        # read from csv file
        images, labels = [], []
        with open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
                
        assert len(images) == len(labels)
        return images, labels
