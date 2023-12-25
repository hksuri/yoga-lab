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
from datetime import datetime
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
from train import run_train
from test import run_test
from dataloader import EyeData

def main():

    # Setting the environment variable to store pretrained model
    # os.environ['TORCH_HOME'] = 'models\\resnet'
    os.environ['TORCH_HOME'] = '/mnt/ssd_4tb_0/huzaifa/models/resnet'

    #GPU CHECK
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU...')
        device = torch.device('cpu')
    else:
        print('CUDA is available. Training on GPU...')
        device = torch.device('cuda:0')

    seed = 23
    seed_everything(seed)

    # Transformations
    trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomRotation((-360, 360)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor()
                                    ])

    # Get dataset
    distorted_data = EyeData(data    = '/mnt/ssd_4tb_0/huzaifa/retina_kaggle/distorted_v2.csv', 
                        root_dir   = '/mnt/ssd_4tb_0/huzaifa/retina_kaggle/resized_train_cropped/label_0_v2',
                        transform  = trans,
                        itype      = '.jpeg')

    # Data loader
    data_len = ( len(os.listdir('/mnt/ssd_4tb_0/huzaifa/retina_kaggle/resized_train_cropped/label_0_v2/v2_distorted')) +
                len(os.listdir('/mnt/ssd_4tb_0/huzaifa/retina_kaggle/resized_train_cropped/label_0_v2/v2')) )

    train_ratio = 0.6
    val_ratio = 0.2

    train_len = int(data_len*train_ratio)
    val_len = int(data_len*val_ratio)
    test_len = data_len - train_len - val_len

    train_set,val_set,test_set = torch.utils.data.random_split(distorted_data,[train_len,val_len,test_len])

    distorted_loader_train = DataLoader(dataset=train_set, batch_size=batch_size_dr, shuffle=True, num_workers=2)
    distorted_loader_val = DataLoader(dataset=val_set, batch_size=batch_size_dr, shuffle=True, num_workers=2)
    distorted_loader_test = DataLoader(dataset=test_set, batch_size=batch_size_dr, shuffle=False, num_workers=2)

    # Initialize model
    net = initialize_model(num_classes_dr, freeze_layers=False, feature_extract=False, use_pretrained=True)
    # num_features = net_gb.fc.in_features
    # net.fc = nn.Sequential(
    #             nn.Dropout(0.5),
    #             nn.Linear(num_features, num_classes_gb))
    net = net.to(device)

    # Print all the trainable parameters
    params_to_update = print_params(net)

    # Print network
    # print(net)

    # Defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, weight_decay = weight_decay_dr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=1e-3, cycle_momentum=False)

    # Train and validate
    train_loss, train_acc, val_loss, val_acc = run_train(net, distorted_loader_train, distorted_loader_val,
                                                         criterion, optimizer, scheduler, num_epochs=num_epochs_dr,
                                                         decay_epochs=10,init_lr=lr_init_dr, task='distort',print_freq=100)

    # Get date and time of model training/validation
    now = datetime.now()
    formatted_date_time = now.strftime("%m_%d_%y_%H_%M")

    # Save model
    model_save_path = f'/mnt/ssd_4tb_0/huzaifa/models/resnet_distort_{formatted_date_time}.pth'
    torch.save(net.state_dict(), model_save_path)

    # Plot and save loss and acc
    plot_loss_acc(train_loss,train_acc,val_loss,val_acc,f'/mnt/ssd_4tb_0/huzaifa/plots/distort_train_val_{formatted_date_time}.png')

    # Test the model on the test set
    net_test = initialize_model(num_classes=2, freeze_layers=True, feature_extract=False, use_pretrained=True)
    net_test.load_state_dict(torch.load(model_save_path))
    net_test.eval()
    net_test = net_test.to(device)

    label, pred, loss, acc = run_test(net_test, distorted_loader_test, criterion)

    # Get AUROC on test set
    get_auroc(label,pred,f'/mnt/ssd_4tb_0/huzaifa/plots/distort_auroc_{formatted_date_time}.png')

    return 0

if __name__ == "__main__":
    main()