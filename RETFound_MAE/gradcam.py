import cv2
import numpy as np
import torch

from pytorch_grad_cam.pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.pytorch_grad_cam.ablation_layer import AblationLayerVit


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def run_cam(model, image_path, args, method='gradcam'):
    
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model.eval()

    if args.device == 'cuda':
        use_cuda = True
    else:
        use_cuda = False

    target_layers = [model.blocks[-1].norm1]

    if method not in methods:
        raise Exception(f"Method {method} not implemented")

    if method == "ablationcam":
        cam = methods[method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=use_cuda,
                                   reshape_transform=reshape_transform)

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.318, 0.154,0.073],
                                    std=[0.146,0.081,0.057])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    # cv2.imwrite(f'{method}_cam.jpg', cam_image)

    return cam_image