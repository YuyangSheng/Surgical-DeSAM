# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
# from engine import evaluate, train_one_epoch
from engine_instance_seg import evaluate, train_one_epoch
from models import build_model
from models.detr_seg import SAMModel
from datasets.coco import *
from segment_anything import sam_model_registry, SamPredictor


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--model', type=str, default=True,
                        help="Whether use the pretrained model for training")
    # * Backbone
    parser.add_argument('--backbone', default='swin_B_224_22k', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone_dir', default='./swin_backbone', type=str,
                        help="The directory of swin transformer backbone")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=20, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.01, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='endovis18')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--endovis_path', type=str, default='./endovis18')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./new_outputs_desam',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # sam parameters
    parser.add_argument('--sam', default=True, type=bool,
                        help='whether do segmentation using Segment Anything Model')
    parser.add_argument('--detr_weights', default='./new_outputs_detr/ckpt_best.pth',
                        help='The directory of well-trained weights of detection model')

    # visualization
    parser.add_argument('--plot', default=True, type=bool,
                        help='whether to visualize results')
    
    return parser


def main(args):
    # device_id = 3  # Index of the GPU you want to use
    # torch.cuda.set_device(device_id)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, seg_criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.sam:
        seg_model = SAMModel(device=device).to(device) # use pre-trained SAM vit-b model
        # Load DETR model for detection part
        checkpoint = torch.load(args.detr_weights, map_location='cpu')
        detr_weights = checkpoint['model']
        # exclude_keys = ['input_proj.weight']
        # for key in exclude_keys:
        #     del detr_weights[key]
        model_without_ddp.load_state_dict(detr_weights)
        train_criterion = seg_criterion # including segmentation and detection losses
        val_criterion = seg_criterion
    else:
        seg_model = None
        train_criterion = criterion
        val_criterion = criterion

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        # base_ds = get_coco_api_from_dataset(dataset_val)
        base_ds = EndovisCOCO(mode='val', dataset=dataset_val)
        # base_ds = None

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        
        if args.sam:
            seg_model.load_state_dict(checkpoint['seg_model'])

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    

    if args.eval:
        epoch = args.start_epoch

        for i in range(61):
            model_path = f'./new_outputs_desam/ckpt{i:04}.pth'
            checkpoint = torch.load(model_path, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            epoch = i

            test_stats, coco_evaluator = evaluate(model_without_ddp, seg_model, val_criterion, 
                                                    postprocessors, data_loader_val, base_ds, 
                                                    device, args.output_dir, epoch, args.plot)
            test_log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                    with (output_dir / "test_log.txt").open("a") as f:
                        f.write(json.dumps(test_log_stats) + "\n")

            if args.output_dir:
                    utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return
    
    print("Start training")
    start_time = time.time()
    best_map = 0
    best_epoch = 0

    # multitask training strategy
    best_metric = 0
    threshold_miou = 0.75
    threshold_map = 0.6

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model_without_ddp, seg_model, train_criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'ckpt_latest.pth']
            # checkpoint_paths = [output_dir / f'ckpt{epoch:04}.pth']
            # # extra checkpoint before LR drop and every 100 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
            #     checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_lr{args.lr}.pth')

        test_stats, coco_evaluator = evaluate(
            model_without_ddp, seg_model, val_criterion, postprocessors, data_loader_val, base_ds, 
            device, args.output_dir, epoch, args.plot
        )
        
        # Find best metric for segmentation
        if args.sam:
            miou= test_stats['coco_eval_masks'][0]
            map_50 = test_stats['coco_eval_bbox'][1]
            curr_metric = (miou + map_50) / 2
            if curr_metric > best_metric and miou > threshold_miou and map_50 > threshold_map:
                best_epoch = epoch
                best_metric = curr_metric
                checkpoint_paths.append(output_dir / 'ckpt_best.pth')
        else:
            # Find best MAP for detection model
            map_50 = test_stats['coco_eval_bbox'][1]
            if map_50 > best_map:
                best_epoch = epoch
                best_map = map_50
                checkpoint_paths.append(output_dir / 'ckpt_best.pth')
        
        if args.output_dir:
            for checkpoint_path in checkpoint_paths:
                if args.sam:
                    utils.save_on_master({
                        'seg_model':seg_model.state_dict(),
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        # 'seg_model':seg_model.state_dict(),
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'best_epoch': best_epoch}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
