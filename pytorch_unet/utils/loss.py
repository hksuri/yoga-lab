import torch
import torch.nn.functional as F

# def ssim(preds, targets, C1=0.01**2, C2=0.03**2):
#     """
#     Computes the Structural Similarity Index (SSIM) between predictions and targets.
#     """
#     # Calculate luminance
#     mu_x = preds.mean([2, 3])
#     mu_y = targets.mean([2, 3])
    
#     # Calculate contrast
#     sigma_x = preds.var([2, 3])
#     sigma_y = targets.var([2, 3])
    
#     # Calculate structure
#     sigma_xy = ((preds - mu_x.unsqueeze(-1).unsqueeze(-1)) * (targets - mu_y.unsqueeze(-1).unsqueeze(-1))).mean([2, 3])
    
#     # Calculate SSIM
#     ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
#     return ssim

def logcosh(preds, targets):
    """
    Computes the Log-cosh loss between predictions and targets.
    """
    return torch.mean(torch.log(torch.cosh(preds - targets)))

def wssl_loss(preds, targets, alpha=0.84):
    """
    Computes the total loss as a weighted sum of Log(cosh) loss and (1 - SSIM) loss.
    """
    # Compute SSIM
    ssim_val = ssim(preds, targets)
    
    # Compute Log(cosh)
    logcosh_val = logcosh(preds, targets)
    
    # Compute the total loss
    loss = alpha * logcosh_val + (1 - alpha) * (1 - ssim_val.mean())
    return loss