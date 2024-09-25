# import sys
# sys.path.append('..')

import os
import os
import torch
import json
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms
from collections import Counter
from dataset.augmenter import AugCompose, RandomHorizontalFlip, RandomVerticalFlip, ImageResize
from dataset.augmenter import RandomRotation, RondomCrop, RondomBrightness, RondomContrast


def transform_type(args, mode):
    if mode == 'Train':
        transforms_op = AugCompose([RandomHorizontalFlip(prob=0.5),
                                    RandomVerticalFlip(prob=0.5),
                                    RandomRotation(limit=30, prob=0.5),
                                    RondomCrop(limit=30, prob=0.5),
                                    RondomBrightness(limit=0.2, prob=0.3),
                                    RondomContrast(limit=0.2, prob=0.3),
                                    ImageResize(args.input_size)])
    else:
        transforms_op = AugCompose([ImageResize(args.input_size)])
    return transforms_op

class _base_folder(torch.utils.data.Dataset):
    def __init__(self, sample_list, domian_num, transform):
        self.samples = sample_list
        self.labellist = [x[-2] for x in sample_list]
        self.set_num = domian_num
        self.transform = transform

    def __getitem__(self, index):
        imgpth = self.samples[index][0]
        mskpth = self.samples[index][1]
        setseq = self.samples[index][2]
        setnam = self.samples[index][3]

        IMG = Image.open(imgpth).convert('RGB')
        MSK = Image.open(mskpth).convert('1')

        IMG, MSK = self.transform(IMG, MSK)
        IMG = transforms.ToTensor()(IMG)
        MSK = transforms.ToTensor()(MSK)

        MK2 = torch.zeros([self.set_num,IMG.shape[1],IMG.shape[2]])
        MK2[setseq,:,:] = MSK

        return IMG, MSK, MK2, setseq, setnam

    def __len__(self):
        return len(self.samples)

def baseloader(args):
    file = open(args.data_configuration, 'r')
    _data_configuration = yaml.load(file.read(), Loader=yaml.FullLoader)
    file.close

    args.domian_num = len(_data_configuration)

    _sample_path_list = {'Train':[],'Valid':[],'Test':[]}

    for _dataset_name, _data_info in _data_configuration.items():
        _single_dataset_list = json.load(open(os.path.join(args.data_path, _data_info['Dataset Cohort']), encoding='utf8', errors='ignore'))
        for _set_name, _set_path_info in _single_dataset_list.items():
            for _set_path in _set_path_info:
                _sample_path_list[_set_name].append([os.path.join(args.data_path,_data_info['Dataset Path'], _set_path['USImage']),
                                                     os.path.join(args.data_path,_data_info['Dataset Path'], _set_path['Mask']),
                                                     _data_info['Dataset Sequence'],
                                                     _dataset_name])

    args.sample_path_list = _sample_path_list

    _tn_folder = _base_folder(sample_list=args.sample_path_list['Train'], domian_num=args.domian_num,
                              transform=transform_type(args, 'Train'))
    _vd_folder = _base_folder(sample_list=args.sample_path_list['Valid'], domian_num=args.domian_num,
                              transform=transform_type(args, 'Valid'))
    _tt_folder = _base_folder(sample_list=args.sample_path_list['Test'], domian_num=args.domian_num,
                              transform=transform_type(args, 'Test'))

    labeldict = dict(Counter(_tn_folder.labellist))
    labellist = torch.tensor([v for k, v in labeldict.items()])
    weight = torch.max(labellist) / labellist.float()
    samples_weight = np.array([weight[t] for t in _tn_folder.labellist])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    args.sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

    _tn_loader = torch.utils.data.DataLoader(_tn_folder, batch_size=args.batch_size, sampler=args.sampler,
                                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                                             drop_last=True)
    _vd_loader = torch.utils.data.DataLoader(_vd_folder, batch_size=args.batch_size,
                                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                                             drop_last=False)
    _tt_loader = torch.utils.data.DataLoader(_tt_folder, batch_size=args.batch_size,
                                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                                             drop_last=False)

    return _tn_loader, _vd_loader, _tt_loader





