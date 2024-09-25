#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24

@author: Haobo CHEN
"""

import os
import warnings
import argparse
import numpy as np
import torch.nn as nn
import torch.multiprocessing
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset.mutlidomain_baseloader import baseloader
from model.MOFO import MOFO
from prior_model.pUNet import pUNet
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss import SelectedDSCLoss, SelectedFLoss
from utils.metric import metric_pixel_dice

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--device', help='device')
parser.add_argument('--num_device', default='1', help='number of device')
parser.add_argument('--log_flag', default=True, help='flag of log')
parser.add_argument('--save_flag', default=True, help='flag of save model')
parser.add_argument('--log_name', default='MOFO', help='name of log')
parser.add_argument('--batch_size', default=28, help='batch size')
parser.add_argument("--epoch", default=0)
parser.add_argument('--max_epoch', default=200, type=int, help='Number of training epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--warmup_epoch', default=1, type=int, help='number of warmup epochs')
parser.add_argument('--weight_decay', default=1e-6, help='Weight Decay')
parser.add_argument('--num_workers', default=12, type=int, help='workers numebr for DataLoader')
parser.add_argument('--input_size', default=(224, 224), help='input size')
parser.add_argument('--log_prior_path', default='prior_weights/prior_weight_sel.pth', help='weight of prior')
parser.add_argument('--data_path', default='Multi-Organ Database/',
                    help='path of the dataset')
parser.add_argument('--data_configuration', default='Multi-Organ Database/dataset_config.yaml',
                    help='configuration of the dataset')
args = parser.parse_args()

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = args.num_device
args.device = torch.device("cuda")

if args.log_flag:
    os.makedirs(os.path.join('output/',args.log_name,'log/'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join('output/',args.log_name,'log/'))
    print('Writing Tensorboard logs to output/{}/log/'.format(args.log_name))

_tn_loader, _vd_loader, _tt_loader = baseloader(args)

model = MOFO(class_num=10, task_prompt='word_embedding')
model.load_from(pretrained_path='model/cswin_small_224.pth')
word_embedding = torch.load('task_weights/prompt_encoding.pth').to(args.device)
model.organ_embedding.data = word_embedding.float()
model.to(args.device)

prior_model = pUNet(n_class=1)
prior_model.load_state_dict(torch.load(args.log_prior_path))
prior_model.to(args.device)

loss_seg_DC = SelectedDSCLoss().to(args.device)
loss_seg_FL = SelectedFLoss().to(args.device)
loss_cls_CE = nn.CrossEntropyLoss().to(args.device)
loss_pri_CL = nn.MSELoss().to(args.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

best_score = 0

while args.epoch < args.max_epoch:
    scheduler.step()
    prior_model.eval()
    model.train()
    loss_seg_dc_ave_tn = 0
    loss_seg_fl_ave_tn = 0
    loss_cls_ce_ave_tn = 0
    loss_con_cl_ave_tn = 0

    epoch_tn_iterator = tqdm(_tn_loader)
    for step, (IMG, MSK1ch, MSK, setseq, _) in enumerate(epoch_tn_iterator):
        IMG, MSK1ch, MSK, setseq = IMG.to(args.device), MSK1ch.to(args.device), MSK.to(args.device), setseq.to(args.device)

        mask_prob_maps, classification_prob_maps = model(IMG)
        mask_pred_maps = torch.sigmoid(mask_prob_maps)
        mask_pred_maps = torch.where(mask_pred_maps > 0.5, 1., 0)
        _, _, prior_info_pred = prior_model(torch.stack([mask_pred_maps[x,setseq[x],:,:] for x in range(args.batch_size)]).unsqueeze(1))
        _, _, prior_info_targ = prior_model(MSK1ch)

        term_seg_DC = loss_seg_DC.forward(mask_prob_maps, MSK, setseq)
        term_seg_FL = loss_seg_FL.forward(mask_prob_maps, MSK, setseq)
        term_cls_CE = loss_cls_CE.forward(classification_prob_maps, setseq)
        term_pri_CL = loss_pri_CL.forward(prior_info_targ, prior_info_pred)

        loss = term_seg_DC + term_seg_FL*0.5 + term_cls_CE*0.3 + term_pri_CL*0.2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_seg_dc_ave_tn += term_seg_DC.item()
        loss_seg_fl_ave_tn += term_seg_FL.item()
        loss_cls_ce_ave_tn += term_cls_CE.item()
        loss_con_cl_ave_tn += term_pri_CL.item()

    loss_seg_dc_tn = loss_seg_dc_ave_tn / len(epoch_tn_iterator)
    loss_seg_fl_tn = loss_seg_fl_ave_tn / len(epoch_tn_iterator)
    loss_cls_ce_tn = loss_cls_ce_ave_tn / len(epoch_tn_iterator)
    loss_con_cl_tn = loss_con_cl_ave_tn / len(epoch_tn_iterator)

    print('Train Epoch=%d: seg_dc_loss=%.5f, seg_fl=%.5f, cls_ce=%.5f, con_cl=%.5f' % (
        args.epoch, loss_seg_dc_tn, loss_seg_fl_tn, loss_cls_ce_tn, loss_con_cl_tn))

    model.eval()
    loss_seg_dc_ave_vd = 0
    loss_seg_fl_ave_vd = 0
    loss_cls_ce_ave_vd = 0
    dsc_list_vd = []
    dsc_dict_vd = dict()

    epoch_vd_iterator = tqdm(_vd_loader)
    for step, (IMG, _, MSK, setseq, uslabel) in enumerate(epoch_vd_iterator):
        IMG, MSK, setseq = IMG.to(args.device), MSK.to(args.device), setseq.to(args.device)

        mask_prob_maps, classification_prob_maps = model(IMG)
        term_seg_DC = loss_seg_DC.forward(mask_prob_maps, MSK, setseq)
        term_seg_FL = loss_seg_FL.forward(mask_prob_maps, MSK, setseq)
        term_cls_CE = loss_cls_CE.forward(classification_prob_maps, setseq)

        loss_seg_dc_ave_vd += term_seg_DC.item()
        loss_seg_fl_ave_vd += term_seg_FL.item()
        loss_cls_ce_ave_vd += term_cls_CE.item()

        _loss_list_vd = metric_pixel_dice(mask_prob_maps, MSK, setseq)
        for k, v in zip(uslabel, _loss_list_vd):
            if not k in dsc_dict_vd.keys():
                dsc_dict_vd[k] = []
            dsc_dict_vd[k].append(v)

    loss_seg_dc_vd = loss_seg_dc_ave_vd / len(epoch_vd_iterator)
    loss_seg_fl_vd = loss_seg_fl_ave_vd / len(epoch_vd_iterator)
    loss_cls_ce_vd = loss_cls_ce_ave_vd / len(epoch_vd_iterator)

    _valid_average_dsc = []
    print('Valid Epoch=%d: seg_dc_loss=%.5f, seg_fl=%.5f, cls_ce=%.5f' % (
        args.epoch, loss_seg_dc_vd, loss_seg_fl_vd, loss_cls_ce_vd))
    for k, v in dsc_dict_vd.items():
        print('Valid {} ave_dice={:.5f} {:.5f}'.format(k, np.mean(v), np.std(v)))
        _valid_average_dsc.append(np.mean(v))

    with open(os.path.join('output/', args.log_name, '_result_valid.txt'), 'a') as f:
        for i in [str(args.epoch)]+['{}-{:.5f}-{:.5f}'.format(k,np.mean(v),np.std(v)) for k, v in dsc_dict_vd.items()]:
            f.write(i + ' ')
        f.write('\n')

    if args.log_flag:
        writer.add_scalar('train/seg_dc', loss_seg_dc_tn, args.epoch)
        writer.add_scalar('train/seg_fl', loss_seg_fl_tn, args.epoch)
        writer.add_scalar('train/cls_ce', loss_cls_ce_tn, args.epoch)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], args.epoch)

        writer.add_scalar('valid/seg_dc', loss_seg_dc_vd, args.epoch)
        writer.add_scalar('valid/seg_fl', loss_seg_fl_vd, args.epoch)
        writer.add_scalar('valid/cls_ce', loss_cls_ce_vd, args.epoch)
        for k, v in dsc_dict_vd.items():
            writer.add_scalar('valid/dice/{}'.format(k), np.mean(v), args.epoch)

    if args.save_flag and args.epoch >= args.warmup_epoch and np.mean(_valid_average_dsc) >= best_score:
        best_score = np.mean(_valid_average_dsc)
        os.makedirs(os.path.join('output/', args.log_name, 'saved_model/'), exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join('output/', args.log_name, 'saved_model/model_epoch_{}.pth'.format(args.epoch)))

    args.epoch += 1



