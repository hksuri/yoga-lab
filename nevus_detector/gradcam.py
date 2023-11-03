import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage.io import imread
from skimage.transform import resize
from utils import get_model

def plot_gradcam(best_model_weights, test_loader, save_dir, threshold=0.5):

    # Load model
    model = get_model(pretrained=best_model_weights)
    
    # Create a 4x4 subplot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    # Loop through selected images and plot them
    for i,(img,label,_) in enumerate(test_loader):

        if i == 16:
            break

        out, acts = model(img)
        predicted_label = torch.argmax(out, dim=1)

        acts = acts.detach()

        loss = nn.CrossEntropyLoss()(out,label.long())
        loss.backward()

        grads = model.get_act_grads().detach()

        pooled_grads = torch.mean(grads, dim=[0,2,3]).detach()

        for i in range(acts.shape[1]):
            acts[:,i,:,:] *= pooled_grads[i]

        heatmap_j = torch.mean(acts, dim = 1).squeeze()
        heatmap_j_max = heatmap_j.max(axis = 0)[0]
        heatmap_j /= heatmap_j_max

        heatmap_j = torch.where(heatmap_j > 0.5, heatmap_j, torch.zeros_like(heatmap_j)).detach().numpy()
        heatmap_j = resize(heatmap_j,(224,224),preserve_range=True)

        # Create a mask for values greater than the threshold
        mask = heatmap_j > threshold
        alpha = mask.astype(float)

        # Apply the mask to the heatmap
        result_heatmap = np.copy(heatmap_j)
        result_heatmap[~mask] = 0

        # Overlay the adjusted heatmap on the image
        combined = img[0].numpy().transpose((1,2,0)).copy()
        combined = combined[:, :, :3]

        # Display the result
        ax = axes[i // 4, i % 4]
        ax.axis('off')

        ax.set_title(f'Prediction: {int(predicted_label[0])}', color='green' if predicted_label == label else 'red')
        ax.imshow(combined)
        ax.imshow(result_heatmap, cmap='jet', alpha=0.2*alpha)  # Use alpha to control the transparency

    # Save the subplot to the specified directory
    fig.tight_layout()
    plt.savefig(save_dir)
    plt.close()
    print(f'\nPreds saved to {save_dir}')

    pass