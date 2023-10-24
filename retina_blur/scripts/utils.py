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

from config import *

def distort(img):
    
    # Motion blur
    blur = [45,60,75,90]
    kernel_size = random.choice(blur)  # Increase for stronger blur
    kernel_direction = np.random.rand(2) - 0.5
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((kernel_size/2-0.5, kernel_size/2-0.5), np.arctan2(kernel_direction[1],
                                                             kernel_direction[0])*180/np.pi, 1), (kernel_size, kernel_size))
    kernel = kernel / kernel_size
    motion_blur = cv2.filter2D(img, -1, kernel)

    # Camera glares
    num_glares = random.randint(0,1)  # Increase for more glares
    glare_size = 250  # Increase for larger glares
    for i in range(num_glares):
        m1 = motion_blur.shape[1] - glare_size
        m2 = motion_blur.shape[0] - glare_size
        if m1 > 0 and m2 > 0:
            x = np.random.randint(0, m1)
            y = np.random.randint(0, m2)
            a = np.random.randint(glare_size-50, glare_size+50)
            b = np.random.randint(glare_size-100, glare_size)
            theta = np.random.randint(0, 360)
            cv2.ellipse(motion_blur, (x,y), (a,b), theta, 0, 360, (255,255,255), -1)

    # Random crop
    # crop_dim = [0.6,0.7,0.8,0.9]
    # distorted = random_crop(motion_blur, size = (random.choice(crop_dim),random.choice(crop_dim)))
    distorted = motion_blur
    
    return distorted

def modify_images(img,
                  input_folder = '/mnt/ssd_4tb_0/huzaifa/retina_kaggle/resized_train_cropped/label_0/v2',
                  output_folder = '/mnt/ssd_4tb_0/huzaifa/retina_kaggle/resized_train_cropped/label_0/v2_distorted'):

    # Define input and output folders
    # input_folder  = '/mnt/ssd_4tb_0/huzaifa/retina_kaggle/resized_train_cropped/label_0/resized_train_cropped_0_label'

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over input images
    for filename in os.listdir(input_folder):
        # Load image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
    
        # Modify image
        modified_img = distort(img)
    
        # Generate output filename
        output_filename = filename[:-5] + '_distorted' + filename[-5:]
        # output_filename = filename
        output_path = os.path.join(output_folder, output_filename)
        
        # Save modified image
        cv2.imwrite(output_path, modified_img)

def move_images(source_folder, destination_folder, csv_file):
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)

    # Extract the filenames from the first column of the DataFrame
    filenames = df.iloc[:, 0]

    # Iterate through each filename
    for filename in filenames:
        # Check if the file exists in the source folder
        src_file = os.path.join(source_folder, filename+'_distorted.jpeg')
        if os.path.exists(src_file):
            # Move the file to the destination folder
            dst_file = os.path.join(destination_folder, filename+'_distorted.jpeg')
            shutil.move(src_file, dst_file)

######### credit for following functions: https://www.kaggle.com/code/sayedmahmoud/diabetic-retinopathy-detection #########

def seed_everything(seed = 23):
    # tests
    assert isinstance(seed, int), 'seed has to be an integer'
    
    # randomness
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_image(path,
                  root_dir,
                  sigmaX         = 10, 
                  do_random_crop = False):
    
    '''
    Preprocess image
    '''
    
    # import image
    image = cv2.imread(os.path.join(root_dir,path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # perform smart crops
    image = crop_black(image, tol = 7)
    if do_random_crop == True:
        image = random_crop(image, size = (0.9, 1))
    
    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    
    # circular crop
    image = circle_crop(image, sigmaX = sigmaX)

    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image

def prepare_image_goodbad(path,
                          root_dir,
                          sigmaX         = 10, 
                          do_random_crop = False):
    
    '''
    Preprocess image
    '''
    
    # import image
    image = cv2.imread(os.path.join(root_dir,path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # perform crops
    if image.shape[1] == 2560:
        image = image[75:1365, 635:1925, :]
    elif image.shape[1] == 2160:
        image = image[:, 360:1800, :]
    elif image.shape[1] == 2256:
        image = image[155:, 484:1774, :]
    
    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    
    # circular crop
    # image = circle_crop(image, sigmaX = sigmaX)

    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image

def crop_black(img, 
               tol = 7):
    
    '''
    Perform automatic crop of black areas
    '''
    
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        
        if (check_shape == 0): 
            return img 
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img  = np.stack([img1, img2, img3], axis = -1)
            return img
        
def circle_crop(img, 
                sigmaX = 10):   
    
    '''
    Perform circular crop around image center
    '''
        
    height, width, depth = img.shape
    
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape
    
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness = -1)
    
    img = cv2.bitwise_and(img, img, mask = circle_img)
    return img 

def random_crop(img, 
                size = (0.9, 1)):
    
    '''
    Random crop
    '''

    height, width, depth = img.shape
    
    cut = 1 - random.uniform(size[0], size[1])
    
    i = random.randint(0, int(cut * height))
    j = random.randint(0, int(cut * width))
    h = i + int((1 - cut) * height)
    w = j + int((1 - cut) * width)

    img = img[i:h, j:w, :]    
    
    return img

###########################################################################################################################

# Function to compute mean and st dev of batches

def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    i = 0

    for images, _ in loader:
        print(i)
        i+=1
        if i == 10:
            break
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)       
     
    return mean,std

def adjust_learning_rate(optimizer, epoch, init_lr, decay_epochs=30):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = init_lr * (0.1 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Function to set last 2 layers trainable, freeze rest of the model
def set_parameter_requires_grad(model):
  
  for name, module in model.named_children():
    if name in ['layer4','fc']:
        for _, para in module.named_parameters():
            para.requires_grad = True
    else:
        for _, para in module.named_parameters():
            para.requires_grad = False

def initialize_model(num_classes, freeze_layers=True, feature_extract=False, use_pretrained=True):

    """
    Initialize the model for this run
    Inputs:
        num_classes: The number of classes that the model will be trained on
        freeze_layers: If True, unfreeze the last two layers and freeze the rest 
        feature_extract: Whether or not to use feature extraction or fine-tuning. Default is False, we fine tune the whole model
        use_pretrained: Whether or not to use a pretrained model
    Outputs:
        model_ft: The model to be used as per the desired number of output classes
    """

    model_ft = None
    model_ft = torchvision.models.resnet18(pretrained = use_pretrained)
    if freeze_layers is True:
        set_parameter_requires_grad(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    return model_ft

def print_params(net):

    print("Params to learn:")
    params_to_update = []
    for name,param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    return params_to_update

def get_auroc(label, pred, save_dir):

    # Get ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(torch.Tensor.cpu(label), torch.Tensor.cpu(pred), pos_label=1)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    # Save the plot to the specified directory
    plt.savefig(save_dir)

    # Print AUROC
    auroc = torchmetrics.AUROC(task="binary")
    a = auroc(pred,label)
    print(f'AUROC: {a}')

def plot_loss_acc(train_loss, train_acc, val_loss, val_acc, save_dir):

    # Initialize plots
    fig,ax = plt.subplots(1,2,figsize=(15,5))

    # Set titles and axis labels
    ax[0].set_title("Training / Validation Loss")
    ax[1].set_title("Training / Validation Accuracy")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Loss")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Accuracy")

    # Plot
    ax[0].plot(val_loss,label="val")
    ax[0].plot(train_loss,label="train")
    ax[1].plot(val_acc,label="val")
    ax[1].plot(train_acc,label="train")

    # Show legends
    ax[0].legend()
    ax[1].legend()

    # Save plot
    plt.savefig(save_dir)