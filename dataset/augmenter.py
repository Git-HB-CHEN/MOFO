#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:04:32 2021

@author: Chnn
"""


import random
from PIL import Image, ImageEnhance

class ImageResize(object):
    def __init__(self, limit):
        self.limit = limit
    def __call__(self, img, msk):
        img = img.resize(self.limit, Image.BICUBIC)
        msk = msk.resize(self.limit, Image.NEAREST)
        return img,msk
        
class RandomHorizontalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob
    def __call__(self, img, msk):
        if random.random() < self.prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
        return img,msk

class RandomVerticalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob
    def __call__(self, img, msk):
        if random.random() < self.prob:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            msk = msk.transpose(Image.FLIP_TOP_BOTTOM)
        return img,msk
            
class RandomRotation(object):
    def __init__(self, limit, prob=0.5):
        self.prob = prob
        self.limit = limit
    def __call__(self, img, msk):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
            img = img.rotate(angle, Image.BICUBIC)
            msk = msk.rotate(angle, Image.NEAREST)
        return img,msk

class RondomCrop(object):
    def __init__(self, limit, prob=0.5):
        self.limit = limit
        self.prob = prob
    def __call__(self, img, msk):
        if random.random() < self.prob:
            x0 = random.randint(0,self.limit)
            y0 = random.randint(0,self.limit) 
            
            x1 = img.size[0] - random.randint(0,self.limit)
            y1 = img.size[1] - random.randint(0,self.limit)
            
            crop_region = [x0,y0,x1,y1]
            img = img.crop(crop_region)
            msk = msk.crop(crop_region)
        return img,msk

class RondomBrightness(object):
    def __init__(self, limit, prob=0.5):
        self.limit = limit
        self.prob = prob
    def __call__(self, img, msk):
        if random.random() < self.prob:
            factor = random.uniform(1-self.limit, 1+self.limit)
            enh_bri = ImageEnhance.Brightness(img)
            img = enh_bri.enhance(factor=factor)
        return img,msk
            
class RondomContrast(object):
    def __init__(self, limit, prob=0.5):
        self.limit = limit
        self.prob = prob
    def __call__(self, img, msk):
        if random.random() < self.prob:
            factor = random.uniform(1-self.limit, 1+self.limit)
            enh_bri = ImageEnhance.Contrast(img)
            img = enh_bri.enhance(factor=factor)
        return img,msk

class AugCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self,x,msk):
        for t in self.transforms:
            x,msk = t(x,msk)          
        return x,msk

    
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# for i in range(1,102):
    
#     Im = Image.open('../../LNs_ALL_Data/ImgUS_Original/Train/方徐星.bmp').convert('RGB') 
#     An = Image.open('../../LNs_ALL_Data/Annotation_Rest/Train/方徐星.bmp').convert('1') 
    
#     transforms_op = AugCompose([
#                                 RandomHorizontalFlip(prob=0.5),
#                                 RandomVerticalFlip(prob=0.5),
#                                 RandomRotation(limit=30,prob=0.5),
#                                 RondomCrop(limit=20,prob=0.5),
#                                 RondomBrightness(limit=0.2,prob=0.5),
#                                 RondomContrast(limit=0.2,prob=0.5),
#                                 ImageResize((256,256))
#                                 ])
#     Im,An = transforms_op(Im,An)
    
#     Im.save('X/{}_X.bmp'.format(str(i)))
#     # An.save('X/{}_y.bmp'.format(str(i)))


