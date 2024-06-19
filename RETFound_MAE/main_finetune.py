# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import csv
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, split_folders, determine_mrn_classes, create_nested_stratified_folds, calculate_weights
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit

from engine_finetune import train_one_epoch, evaluate

from get_unet import UnetEncoderHead
from get_resnet import CustomResNet

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='/research/labs/ophthalmology/iezzi/m294666/retfound_model/RETFound_cfp_weights.pth',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='/research/labs/ophthalmology/iezzi/m294666/retfound_task_dia5/',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--log_task', default='/dia5/',type=str,
                    help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    # parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--rf', type=str, required=True,
                        help = 'risk factor from dia5, intref, orange, va, thick2, srf')
    parser.add_argument('--zoom', type=str, default='',
                        help = 'zoom in or out of the validation images')
    parser.add_argument('--zoom_level', type=float, default=0.,
                        help = 'zoom level')
    parser.add_argument('--save_images', action='store_true', default=False)
    parser.add_argument('--freeze', action='store_true', default=False)
    parser.add_argument('--unet_checkpoint',
                        # default='/research/labs/ophthalmology/iezzi/m294666/unet_files/base_model/best_checkpoint_dr.pth', type=str)
                        default='/research/labs/ophthalmology/iezzi/m294666/unet_files/output/best_checkpoint.pth', type=str)
    parser.add_argument('--resnet_model_name', default='resnet18', type=str)

    # Dataset parameters
    # parser.add_argument('--data_path', default='/home/jupyter/Mor_DR_data/data/data/IDRID/Disease_Grading/', type=str,
    parser.add_argument('--data_path', default='/research/labs/ophthalmology/iezzi/m294666/data_dia5_retfound', type=str,
    # parser.add_argument('--data_path', default='/mnt/ssd_4tb_0/huzaifa/retfound_dummy', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='/research/labs/ophthalmology/iezzi/m294666/retfound_output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/research/labs/ophthalmology/iezzi/m294666/retfound_output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', default=False, action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # attention rollout parameters
    parser.add_argument('--discard_ratio', type=float, default=0.2)
    parser.add_argument('--head_fusion', type=str, default='max')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    test_aurocs = 0.
    true_labels_list = []
    output_prob_list = []
    preds_list = []
    embeddings_lists_main = []
    img_paths_main = []

    folds = 5
    # best_val_auc_across_all = 0.
    # Create train/val/test MRN list
    # train_val_sets, test_set = split_folders(args.data_path)
    # train_val_sets = split_folders(args.data_path)
    mrn_classes = determine_mrn_classes(args.data_path, args.rf)
    folds_list = create_nested_stratified_folds(mrn_classes)
    loss_weights = calculate_weights(mrn_classes).to(device)

    print(f'Weights based on class distribution: {loss_weights}')

    for fold in range(folds):

        # if fold == folds: val = 0
        # else: val = 1
        val = 1

        ###########################################################################

        if val:

            print(f'\n\n######## Fold: {fold+1} #############\n')

            if args.eval:
                # create val_set using all the train_val_sets
                # train_set = [item for sublist in train_val_sets for item in sublist]
                train_set = folds_list[fold][0] + folds_list[fold][1] + folds_list[fold][2]
                print(f'Length of train_set: {len(train_set)}')
                dataset_train = build_dataset(is_train='val', mrn_list=train_set, args=args)
            
            else:
                # train_set = [set_ for i, set_ in enumerate(train_val_sets) if i != fold]
                # train_set = [item for sublist in train_set for item in sublist]

                # # Calculate the size of each part to divide the list into 8 equal parts
                # part_size = len(train_set) // 8
                # # Extract the last 1/8th of the list for the validation set
                # val_set = train_set[-part_size:]
                # # Keep the first 7/8th of the list as the training set
                # train_set = train_set[:-part_size]

                # test_set = train_val_sets[fold]
                train_set, val_set, test_set = folds_list[fold][0], folds_list[fold][1], folds_list[fold][2]
                dataset_train = build_dataset(is_train='train', mrn_list=train_set, args=args)
                dataset_val = build_dataset(is_train='val', mrn_list=val_set, args=args)
                dataset_test = build_dataset(is_train='test', mrn_list=test_set, args=args)

                print(f'Train set size: {len(train_set)}, val set size: {len(val_set)}, test set size: {len(test_set)}\n')

        else:

            print(f'\n\n######## Final Training #############\n')

            # train_set = [item for sublist in train_val_sets for item in sublist]
            # dataset_train = build_dataset(is_train='train', mrn_list=train_set, args=args)

        # dataset_test = build_dataset(is_train='test', mrn_list=test_set, args=args)

        ###########################################################################

        # dataset_train = build_dataset(is_train='train', args=args)
        # dataset_val = build_dataset(is_train='val', args=args)
        # dataset_test = build_dataset(is_train='test', args=args)

        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)
        
        model = models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

        # model = UnetEncoderHead(args.unet_checkpoint, args, n_channels=3, n_classes=3, output_classes=2)

        # os.environ['TORCH_HOME'] = '/research/labs/ophthalmology/iezzi/m294666/base_models'
        # model = CustomResNet(model_name=args.resnet_model_name, num_classes=args.nb_classes)
        # model = models.resnet50(weights = 'DEFAULT')
        # num_features = model.fc.in_features
        # model.fc = nn.Sequential(
        #     nn.Linear(num_features, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 2)  # Output layer for binary classification
        # )

        # if args.distributed:
        if True:  # args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if val and not eval:
                if args.dist_eval:
                    if len(dataset_val) % num_tasks != 0:
                        print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                            'This will slightly alter validation results as extra duplicate entries are added to achieve '
                            'equal num of samples per-process.')
                    sampler_val = torch.utils.data.DistributedSampler(
                        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
                else:
                    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                
            # if args.dist_eval:
                # if len(dataset_test) % num_tasks != 0:
                #     print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                #         'This will slightly alter validation results as extra duplicate entries are added to achieve '
                #         'equal num of samples per-process.')
                # sampler_test = torch.utils.data.DistributedSampler(
                #     dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            # else:
                # sampler_test = torch.utils.data.SequentialSampler(dataset_test)


        print(f'global rank: {global_rank}, num tasks: {num_tasks}') 

        if global_rank == 0 and args.log_dir is not None and not args.eval:
        # if args.log_dir is not None and not args.eval:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir+args.log_task)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        if val and not args.eval:
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            # if args.global_pool:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # # manually initialize fc layer
            # trunc_normal_(model.head.weight, std=2e-5)

        # Freeze every layer except the head
        if args.freeze:
            for name, param in model.named_parameters():
                if 'head1' not in name and 'head2' not in name:
                # if 'head' not in name:
                    param.requires_grad = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        if args.freeze:
            print("Freezing all layers except the head")
            print('number of params: %.2f' % (n_parameters))
        else:
            print('number of params (M): %.2f' % (n_parameters / 1.e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        
        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        loss_scaler = NativeScaler()

        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
        #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        # else:
            criterion = torch.nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.smoothing)

        print("criterion = %s" % str(criterion))

        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        if args.eval:
            print(f"Start evaluation:\n")
            gt,_,pred,_,_,test_stats,auc_roc = evaluate(args, data_loader_train, model, device, args.task, epoch=0, mode='test',num_class=args.nb_classes, save_images=args.save_images)
            auc_roc_all = roc_auc_score(gt, pred,multi_class='ovr',average='macro')
            print(f'\n\nAverage validation AUROC for all images: {auc_roc_all}\n')
            exit(0)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_auc = 0.0
        train_loss = []
        val_loss = []

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            
            # Train
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )

            # Validation
            _,_,_,_,_,val_stats,val_auc_roc = evaluate(args, data_loader_val, model, device,args.task,epoch, mode='val',num_class=args.nb_classes, save_images=False)
            # scheduler.step(val_auc_roc)

            # Save train and val loss
            train_loss.append(train_stats['loss'])
            val_loss.append(val_stats['loss'])

            # Save best model
            if max_auc < val_auc_roc:
                max_auc = val_auc_roc
                misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch)
                print('*** Best model found. Validation AUROC: %.4f ***' % val_auc_roc)
            
            # Test
            if epoch==(args.epochs-1) and val:

                print('*** Testing best validation model on test set ***')

                # Load best model
                checkpoint = torch.load(args.task + 'checkpoint-best.pth', map_location='cpu')
                model_without_ddp.load_state_dict(checkpoint['model'])
                # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

                gt,out_prob,pred,img_paths,embeddings,_,test_auc_roc = evaluate(args, data_loader_test, model, device,args.task,epoch, mode='test',num_class=args.nb_classes, save_images=args.save_images)
                
                true_labels_list.extend(gt)
                output_prob_list.extend(out_prob)
                preds_list.extend(pred)
                embedding_lists = [embedding.tolist() for embedding in embeddings]
                embeddings_lists_main.extend(embedding_lists)
                img_paths_main.extend(img_paths)

                    # _,_,_,_,_,_,_ = evaluate(args, data_loader_val, model, device,args.task,epoch, mode='val',num_class=args.nb_classes, save_images=args.save_images)
            # else:
            #     misc.save_model(
            #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #         loss_scaler=loss_scaler, epoch=epoch)

        # Plot train/val loss
        misc.plot_loss(train_loss, val_loss, fold, args)    

        if val: test_aurocs += test_auc_roc
        if fold == folds-1:
            print(f'\n\nAverage test AUROC: {test_aurocs/folds}\n')
            # print(f'\nLen of true label and pred list: {len(true_labels_list)} and {len(preds_list)}')
            # print(f'\nOne true label: {true_labels_list[0]}')
            # print(f'\nOne pred: {preds_list[0]}')
            # print(f'\nSize of each true label and pred element: {true_labels_list[0].shape} and {preds_list[0].shape}')
            auc_roc_all = roc_auc_score(true_labels_list, preds_list,multi_class='ovr',average='macro')
            print(f'\n\nAverage test AUROC for all images: {auc_roc_all}\n')

    # Save image paths, true labels, output probs, and embeddings to a CSV file
    misc.save_test_data(img_paths_main, true_labels_list, output_prob_list, embeddings_lists_main, args)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
