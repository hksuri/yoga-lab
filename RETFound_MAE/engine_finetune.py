# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix
from pycm import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime

from PIL import Image

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from vit_explain import LRP
from gradcam import run_cam
from mm_explain import generate_visualization

def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_





def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (_, samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # print(f'samples size: {samples.size()}')
            _,outputs = model(samples)
            # outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




# @torch.no_grad()
def evaluate(args, fold, data_loader, model, device, task, epoch, mode, num_class,save_images):
    print(f'Mode: {mode}')
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    ############################################################
    
    # Create new folder named test_images in task. Delete if exists already

    # Get today's year month day
    # today = datetime.date.today()
    # today = today.strftime('%Y%m%d')

    # folder_name = f'test_images_{today}_epochs{args.epochs}_discard{args.discard_ratio}_{args.head_fusion}'
    folder_name = f'test_images_{args.run_date}_epochs{args.epochs}_gradcam'
    if save_images:
    
        if not os.path.exists(task+folder_name):
            os.makedirs(task+folder_name)
        # else:
            # os.system('rm -r '+task+folder_name)
            # os.makedirs(task+folder_name)

        # Define the VITAttentionGradRollout object
        # grad_rollout = VITAttentionRollout(model, discard_ratio=args.discard_ratio, head_fusion=args.head_fusion)
        # grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        # attribution_generator = LRP(model)

    ############################################################

    prediction_decode_list = []
    output_prob_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []

    image_paths = []
    model_embeddings = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        image_paths.extend(batch[0])
        images = batch[1]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                emb,output = model(images)
            model_embeddings.extend(emb)
            # output = model(images)
            # model_embeddings.extend(np.zeros(output.size(0)))
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            max_probs,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

            output_prob_list.extend(prediction_softmax.cpu().detach().numpy())

        if save_images:
            # For all images in a batch, generate the grad_rollout mask, put it on the image and save it in the test_images folder
            for i in range(images.size(0)):

                gt = target[i].item()
                pred = prediction_decode[i].item()
                prob = max_probs[i].item()
                prob = round(prob, 3)
                path = batch[0][i]
                
                img_original = Image.open(path)
                # img_original = img_original.resize((224, 224))
                
                # img_tensor = images[i:i+1]
                # print(f'img_tensor size: {img_tensor.size()}')
                # mask = grad_rollout(img_tensor,1)
                # mask = grad_rollout(img_tensor)
                # transformer_attribution = attribution_generator.generate_LRP(img_tensor, method="transformer_attribution", index=1).detach()
                # transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
                # transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
                # transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
                # mask = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
                # print(f'mask size: {mask.shape}')

                # Resize img_original to 224 x 224
                img_original = img_original.resize((224, 224))
                
                np_img = np.array(img_original)[:, :, ::-1]
                # mask = np.zeros_like(np_img)
                # mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))

                # print(f'image original max, mean, min: {np_img.max()}, {np_img.mean()}, {np_img.min()}')

                img = np.float32(np_img) / 255
                # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                # heatmap = np.float32(heatmap) / 255

                # print(f'heatmap max, mean, min: {heatmap.max()}, {heatmap.mean()}, {heatmap.min()}')

                # cam = heatmap + np.float32(img)
                # cam = cam / np.max(cam)

                cam = run_cam(model, path, args, method='gradcam')
                # cam = generate_visualization(model, path)

                img = np.uint8(255 * img)
                cam = cv2.hconcat([img, cam])
                # cam = np.uint8(255 * cam)
                cv2.putText(cam, f'GT: {gt}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(cam, f'Pred: {pred}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(cam, f'Prob: {prob}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                
                # print(f'cam max, mean, min: {cam.max()}, {cam.mean()}, {cam.min()}')
                
                # Save img_original on the left and cam on the right
                cv2.imwrite(task+folder_name+'/'+str(path).split('/')[-1].split('.')[0]+'_pred.jpg', cam)

        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')          
            
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
    results_path = task+'metrics_{}.csv'.format(mode)
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)

    folder_name = f'loss_{args.run_date}_epochs{args.epochs}'
    if not os.path.exists(args.task+folder_name):
        os.makedirs(args.task+folder_name)
            
    if mode=='test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+folder_name+f'/confusion_matrix_test_{fold}.jpg',dpi=600,bbox_inches ='tight')
    
    return true_label_onehot_list,output_prob_list,prediction_list,image_paths, model_embeddings,{k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc

