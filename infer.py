#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24

@author: Haobo CHEN
"""

import os
import cv2
import warnings
import argparse
import numpy as np
import torch.multiprocessing
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from model.MOFO import MOFO

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--device', help='device')
parser.add_argument('--num_device', default='1', help='number of device')
parser.add_argument('--input_size', default=(224, 224), help='input size')
parser.add_argument('--example_path', default='examples/', help='path of the dataset')
args = parser.parse_args()

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = args.num_device
args.device = torch.device("cuda")

model = MOFO(class_num=10, task_prompt='word_embedding')
model.load_state_dict(torch.load('mofo_weights/mofo_weight_sel.pth'))
model.to(args.device)
model.eval()

for filename in tqdm(os.listdir(args.example_path)):
    if filename != 'Inference':
        imgpth = os.path.join(args.example_path, filename)
        IMG_Ori = Image.open(imgpth).convert('RGB')
        IMG = IMG_Ori.resize(args.input_size, Image.BICUBIC)
        IMG = transforms.ToTensor()(IMG)

        IMG = IMG.unsqueeze(0)
        IMG = IMG.to(args.device)
        mask_prob_maps, classification_prob_maps = model(IMG)
        setseq = classification_prob_maps.argmax().tolist()
        mask_prob_maps = torch.sigmoid(mask_prob_maps)
        mask_pred_maps = torch.where(mask_prob_maps > 0.5, 1., 0)
        mask_pred_map = (np.array(mask_pred_maps[:, setseq, :, :].squeeze().tolist())*255).astype(float)

        mask_pred_map_resize = cv2.resize(mask_pred_map, IMG_Ori.size)
        os.makedirs(os.path.join(args.example_path, 'Inference'), exist_ok=True)
        cv2.imencode('.png', mask_pred_map_resize)[1].tofile(os.path.join(args.example_path, 'Inference', filename))


