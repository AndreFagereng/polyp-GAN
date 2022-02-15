import os
import cv2
import math
import numpy as np
from glob import glob

from random import shuffle
from PIL import Image, ImageFilter

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        self.mask_type = args.mask_type
        
        # image and mask 
        self.image_path = []
        self.mask_path = []
        for ext in ['*.jpg', '*.png']: 
            self.image_path.extend(glob(os.path.join(args.dir_image, args.data_train, ext)))
            self.mask_path.extend(glob(os.path.join(args.dir_mask, args.mask_type, ext)))

        # augmentation 
        self.img_trans = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()])
        self.mask_trans = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                (0, 45), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])
        if self.mask_type == 'pconv' or self.mask_type == 'irregular_mask':
            index = np.random.randint(0, len(self.mask_path))
            mask = Image.open(self.mask_path[index])
            mask = mask.convert('L')
            #print('wrong masks')
            #print(self.image_path[index])
            #print(self.mask_path[index])
        elif self.mask_type == 'masks':
            mask = Image.open(self.mask_path[index])
            #print(self.mask_path[index])
            mask = mask.convert('L') 
        #else:
        #    mask = np.zeros((self.h, self.w)).astype(np.uint8)
        #    mask[self.h//4:self.h//4*3, self.w//4:self.w//4*3] = 1
        #    mask = Image.fromarray(mask).convert('L')

        #index = np.random.randint(0, len(self.mask_path))
        #mask = cv2.imread(self.mask_path[index], cv2.IMREAD_GRAYSCALE)
        #(thresh, mask) = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        #mask = np.asarray(mask)
        # augment
        image = self.img_trans(image) * 2. - 1.
        mask = F.to_tensor(self.mask_trans(mask))
        #print(np.unique(mask.view(-1)))
        #print(mask)
        return image, mask, filename



if __name__ == '__main__': 

    from attrdict import AttrDict
    args = {
        'dir_image': r'C:/Users/Jegern/Masteroppgave/data/kvasir-seq-1000/Kvasir-SEG/',
        'data_train': 'images',
        'dir_mask': r'C:/Users/Jegern/Masteroppgave/data/kvasir-seq-1000/Kvasir-SEG/',
        'mask_type': 'masks',
        'image_size': 512
    }
    args = AttrDict(args)

    data = InpaintingData(args)
    print(len(data), len(data.mask_path))
    img, mask, filename = data[0]
    print(img.size(), mask.size(), filename)