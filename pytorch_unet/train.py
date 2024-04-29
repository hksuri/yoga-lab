import argparse
import logging
# import gc
import os
# import random
# import sys
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
# import numpy as np
# from PIL import Image

# import wandb
# from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
# from utils.dice_score import dice_loss
from utils.utils import plot_img_and_mask, plot_train_val_loss, apply_random_transformations
from utils.loss import wssl_loss
from inpaint import load_freeform_masks, inpaint_freeform

# dir_img = '/mnt/ssd_4tb_0/huzaifa/retina_kaggle/resized_train_cropped/label_0/resized_train_cropped_0_label/'
dir_img = '/research/labs/ophthalmology/iezzi/m294666/unet_files/data'
# dir_img = '/mnt/ssd_4tb_0/huzaifa/retina_kaggle/resized_train_cropped/label_0/test/'
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('/mnt/ssd_4tb_0/huzaifa/unet/checkpoints/')
dir_checkpoint = Path('/research/labs/ophthalmology/iezzi/m294666/unet_files/checkpoints/')
# dir_output = '/home/huzaifa/workspace/Pytorch-UNet/output/'
dir_output = '/research/labs/ophthalmology/iezzi/m294666/unet_files/output/'

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    freeform_masks = load_freeform_masks('freeform1020', args.freeform_dir)
    dataset = BasicDataset(dir_img, freeform_masks)

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    total_size = len(dataset)
    train_percent = 0.70
    val_percent = 0.15
    test_percent = 0.15  # Assuming the rest goes to the test

    n_train = int(total_size * train_percent)
    n_val = int(total_size * val_percent)
    n_test = total_size - n_train - n_val  # Ensuring we use all data

    # Splitting the dataset
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0)
)

    # 3. Create data loaders
    train_loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, **train_loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **test_loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    # criterion = torch.nn.MSELoss(reduction='mean')
    criterion = wssl_loss
    global_step = 0

    train_loss = []
    val_loss = []
    best_val = float('inf')
    best_checkpoint = None

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for img, mask, img_masked, img_name in train_loader:

                # Random transformations for training images
                # img = apply_random_transformations(img)
                # img_masked = img * (1. - mask) + mask
                
                assert img_masked.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {img_masked.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images_masked = img_masked.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = img
                # true_masks = img*mask
                mask = mask.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images_masked)
                    # masks_pred *= mask
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        # loss += dice_loss(
                        #     F.softmax(masks_pred, dim=1).float(),
                        #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #     multiclass=True
                        # )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images_masked.shape[0])
                global_step += 1
                # epoch_loss += loss.item()
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            train_loss.append(epoch_loss/len(train_loader))

        # 6. Test on validation data
        model.eval()
        val_epoch_loss = 0
        for img, mask, img_masked, img_name in val_loader:

            image_masked = img_masked.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # true_mask = img*mask
            true_mask = img
            mask = mask.to(device=device, dtype=torch.float32)
            true_mask = true_mask.to(device=device, dtype=torch.float32)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                mask_pred = model(image_masked)
                # mask_pred *= mask
                loss = criterion(mask_pred, true_mask)
                val_epoch_loss += loss.item()

        avg_val_loss = val_epoch_loss/len(val_loader)
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_checkpoint = model.state_dict().copy()
            # Save model in output directory
            torch.save(best_checkpoint, dir_output + 'best_checkpoint.pth')
        
        val_loss.append(avg_val_loss)

    # 7. Save 10 images
    i = 0
    # Load best checkpoint
    # val_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    # val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **val_loader_args)
    model.load_state_dict(best_checkpoint)
    print(f'\nBest mode loaded with validation loss: {best_val}')
    for img, mask, img_masked, img_name in test_loader:

        i += 1
        if i > 50:
            break

        image_masked = img_masked.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        # true_mask = img*mask
        true_mask = img
        mask = mask.to(device='cpu', dtype=torch.float32)
        true_mask = true_mask.to(device=device, dtype=torch.float32)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            mask_pred = model(image_masked).cpu()
        
        # mask_pred *= mask
        # img_inpainted = img * (1. - mask) + mask_pred
        img_inpainted = mask_pred
        
        plot_img_and_mask(img.detach(), image_masked.detach(), img_inpainted.detach(), mask, img_name, dir_output)

    # 8. Save training and validation loss
    plot_train_val_loss(train_loss, val_loss, dir_output)

                # # Evaluation round
                # division_step = (n_train // (5 * batch_size))
                # if division_step > 0:
                #     if global_step % division_step == 0:
                #         histograms = {}
                #         for tag, value in model.named_parameters():
                #             tag = tag.replace('/', '.')
                #             if not (torch.isinf(value) | torch.isnan(value)).any():
                #                 histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                #             if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                #                 histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                #         val_score = evaluate(model, val_loader, device, amp)
                #         scheduler.step(val_score)

                #         # logging.info('Validation Dice score: {}'.format(val_score))
                #         try:
                #             experiment.log({
                #                 'learning rate': optimizer.param_groups[0]['lr'],
                #                 # 'validation Dice': val_score,
                #                 'images': wandb.Image(images[0].cpu()),
                #                 'masks': {
                #                     'true': wandb.Image(true_masks[0].float().cpu()),
                #                     'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                #                 },
                #                 'step': global_step,
                #                 'epoch': epoch,
                #                 **histograms
                #             })
                #         except:
                #             pass

    #     if save_checkpoint:
    #         Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    #         state_dict = model.state_dict()
    #         state_dict['mask_values'] = dataset.mask_values
    #         torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
    #         logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--freeform_dir', type=str, default='/research/labs/ophthalmology/iezzi/m294666/unet_files', help='Path to freeform masks directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
