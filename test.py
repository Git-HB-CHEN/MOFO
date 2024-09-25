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
import torch.multiprocessing
from tqdm import tqdm
from dataset.mutlidomain_baseloader import baseloader
from model.MOFO import MOFO
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
parser.add_argument('--num_workers', default=12, type=int, help='workers numebr for DataLoader')
parser.add_argument('--input_size', default=(224, 224), help='input size')
parser.add_argument('--data_path', default='Multi-Domain Database/',
                    help='path of the dataset')
parser.add_argument('--data_configuration', default='Multi-Domain Database/dataset_config.yaml',
                    help='configuration of the dataset')
args = parser.parse_args()

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = args.num_device
args.device = torch.device("cuda")

_, _, _tt_loader = baseloader(args)

seleted_epoch = 1
model = MOFO(class_num=10, task_prompt='word_embedding')
model.load_state_dict(torch.load(os.path.join('output/', args.log_name, 'saved_model/model_epoch_{}.pth'.format(seleted_epoch))))
model.to(args.device)
model.eval()

dsc_list_tt = []
dsc_dict_tt = dict()

epoch_tt_iterator = tqdm(_tt_loader)
for step, (IMG, _, MSK, setseq, uslabel) in enumerate(epoch_tt_iterator):
    IMG, MSK,setseq = IMG.to(args.device), MSK.to(args.device), setseq.to(args.device)
    mask_prob_maps, _ = model(IMG)

    _loss_list_tt = metric_pixel_dice(mask_prob_maps, MSK, setseq)
    for k, v in zip(uslabel, _loss_list_tt):
        if not k in dsc_dict_tt.keys():
            dsc_dict_tt[k] = []
        dsc_dict_tt[k].append(v)

Result_Matrix = {}
Result_List = []
for k, v in dsc_dict_tt.items():
    print('Test {} ave_dice={:.5f} {:.5f}'.format(k, np.mean(v), np.std(v)))
    Result_List.append(round(np.mean(v)*100, 2))
    Result_Matrix[k] = round(np.mean(v)*100, 2)

Result_Matrix['Avg.'] = round(np.mean(Result_List), 2)

print(Result_Matrix)




