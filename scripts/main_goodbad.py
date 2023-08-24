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
from scripts.train import run_train
from scripts.test import run_test
from dataloader import GoodBadDataset

MODEL_SAVE_PATH = None

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
    transform_goodbad = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((224,224))
    ])

    data_len = ( len(os.listdir('/mnt/ssd_4tb_0/huzaifa/good_bad_new_v2/images/good')) +
                len(os.listdir('/mnt/ssd_4tb_0/huzaifa/good_bad_new_v2/images/bad')) )

    train_ratio = 0.6
    val_ratio = 0.2

    train_len = int(data_len*train_ratio)
    val_len = int(data_len*val_ratio)
    test_len = data_len - train_len - val_len

    # Get dataset
    dataset = GoodBadDataset(root_dir = '/mnt/ssd_4tb_0/huzaifa/good_bad_new_v2/images',
                            csv_dir = '/mnt/ssd_4tb_0/huzaifa/good_bad_new_v2/images.csv',
                            transform = transform_goodbad)

    train_set_gb,val_set_gb,test_set_gb = torch.utils.data.random_split(dataset,[train_len,val_len,test_len])

    # Get dataloaders
    trainloader_gb = DataLoader(dataset=train_set_gb, batch_size=batch_size_gb, shuffle=True, num_workers=4)
    valloader_gb = DataLoader(dataset=val_set_gb, batch_size=batch_size_gb, shuffle=True, num_workers=4)
    testloader_gb = DataLoader(dataset=test_set_gb, batch_size=batch_size_gb, shuffle=True, num_workers=4)

    # Load model for fine tuning
    net_gb = initialize_model(num_classes=2, freeze_layers=False, feature_extract=False, use_pretrained=True)
    net_gb.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Replace fc layer for good/bad classification
    num_features = net_gb.fc.in_features
    net_gb.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes_gb))

    net_gb = net_gb.to(device)

    # Print all the trainable parameters
    params_to_update = print_params(net_gb)

    # Defining loss function and optimizer
    criterion_gb = nn.CrossEntropyLoss()
    optimizer_gb = optim.Adam(params_to_update, weight_decay = weight_decay_gb)
    scheduler_gb = torch.optim.lr_scheduler.CyclicLR(optimizer_gb, base_lr=1e-8, max_lr=1e-4, cycle_momentum=False)

    # Train and validate
    train_loss, train_acc, val_loss, val_acc = run_train(net_gb,trainloader_gb, valloader_gb, criterion_gb, optimizer_gb, scheduler_gb,
                                                         num_epochs=num_epochs_gb, decay_epochs=10,
                                                         init_lr=lr_init_gb, task='goodbad', print_freq = 2)

    # Get date and time of model training/validation
    now = datetime.now()
    formatted_date_time = now.strftime("%m_%d_%y_%H_%M")

    # Save model
    model_save_path = f'/mnt/ssd_4tb_0/huzaifa/models/resnet_goodbad_{formatted_date_time}.pth'
    torch.save(net_gb.state_dict(), model_save_path)

    # Plot and save loss and acc
    plot_loss_acc(train_loss,train_acc,val_loss,val_acc,f'/mnt/ssd_4tb_0/huzaifa/plots/goodbad_train_val_{formatted_date_time}.png')

    # Test the model on the test set
    net_test = initialize_model(num_classes=2, freeze_layers=True, feature_extract=False, use_pretrained=True)
    net_test.load_state_dict(torch.load(model_save_path))
    net_test.eval()
    net_test = net_test.to(device)

    label, pred, loss, acc = run_test(net_test, testloader_gb, criterion_gb)

    # Get AUROC on test set
    get_auroc(label,pred,f'/mnt/ssd_4tb_0/huzaifa/plots/distort_auroc_{formatted_date_time}.png')

    return 0

if __name__ == "__main__":
    main()