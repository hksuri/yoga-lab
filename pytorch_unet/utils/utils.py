import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def apply_random_transformations(images):
    # Define the transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),  # Adjust the degrees of rotation as needed
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Adjust the parameters of ColorJitter as needed
    ])
    transformed_images = torch.stack([transform(image) for image in images]) 
    return transformed_images


def plot_train_val_loss(train_loss, val_loss, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss.png")
    plt.close()

def plot_img_and_mask(img_org, img_in, img_out, mask, img_name, dir_output, mode = 'inpainted image'):

    file_path = dir_output + str(img_name).strip("[]'").replace('.jpg', '.png')
    
    # Normalizing and converting original image
    img = img_org
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # Normalizing and converting input image
    # image_normalized_in = (img_in + 1) / 2
    image_normalized_in = img_in
    image_normalized_in = image_normalized_in.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_back_in = np.clip(image_normalized_in * 255, 0, 255).astype(np.uint8)

    # Normalizing and converting outut image
    # image_normalized_out = (img_out + 1) / 2
    image_normalized_out = img_out
    image_normalized_out = image_normalized_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_back_out = np.clip(image_normalized_out * 255, 0, 255).astype(np.uint8)

    # Convert numpy arrays to PIL images
    image_back_org_pil = Image.fromarray(img)
    image_back_in_pil = Image.fromarray(image_back_in)
    image_back_out_pil = Image.fromarray(image_back_out)

    # Determine the mode based on user input
    if mode == 'masked input':
        image_back_out_pil = Image.fromarray(image_back_out * mask.transpose(1, 2, 0))

    # Creating a new image with the appropriate dimensions
    width = image_back_org_pil.width + image_back_in_pil.width + image_back_out_pil.width
    height = max(image_back_org_pil.height, image_back_in_pil.height, image_back_out_pil.height)
    combined_img = Image.new('RGB', (width, height))

    # Place image_back_org on the left
    combined_img.paste(image_back_org_pil, (0, 0))

    # Place image_back_in in the middle
    combined_img.paste(image_back_in_pil, (image_back_org_pil.width, 0))

    # Place image_back_out on the right
    combined_img.paste(image_back_out_pil, (image_back_org_pil.width + image_back_in_pil.width, 0))

    # Save the combined image
    combined_img.save(file_path)

    # classes = mask.max() + 1
    # fig, ax = plt.subplots(1, classes + 1)
    # ax[0].set_title('Input image')
    # ax[0].imshow(img)
    # for i in range(classes):
    #     ax[i + 1].set_title(f'Mask (class {i + 1})')
    #     ax[i + 1].imshow(mask == i)
    # plt.xticks([]), plt.yticks([])
    # plt.show()
