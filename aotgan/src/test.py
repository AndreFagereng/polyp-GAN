import os
import argparse
import importlib
import numpy as np
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

from utils.option import args 


def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


def main_worker(args, use_gpu=True): 

    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    print(args)
    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location='cuda'))
    model.eval()

    # prepare dataset
    image_paths = []
    for ext in ['.jpg', '.png']: 
        image_paths.extend(glob(os.path.join(args.dir_image, '*'+ext)))
    image_paths.sort()
    mask_paths = sorted(glob(os.path.join(args.dir_mask, '*.jpg')))
    os.makedirs(args.outputs, exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'images'))
    
    test_img_size = (256, 256)
    # iteration through datasets
    #print(image_paths)
    #print(mask_paths)

    image_paths = image_paths[:500]
    mask_paths = mask_paths[:len(image_paths)]

    for ipath, mpath in zip(image_paths, mask_paths): 
        
        img = Image.open(ipath).convert('RGB').resize(test_img_size)
        msk = Image.open(mpath).convert('L').resize(test_img_size)
        print(msk)
        print(msk.size)
        print(img.size)
        
        image = ToTensor()(img)
        image = (image * 2.0 - 1.0).unsqueeze(0)
        mask = ToTensor()(msk)
        mask = mask.unsqueeze(0)
        image, mask = image.cuda(), mask.cuda()
        image_masked = image * (1 - mask.float()) + mask

        with torch.no_grad():
            pred_img = model(image_masked, mask)
        comp_imgs = (1 - mask) * image + mask * pred_img
        image_name = os.path.basename(ipath).split('.')[0]
        postprocess(image_masked[0]).save(os.path.join(args.outputs, f'{image_name}_masked.png'))
        #postprocess(pred_img[0]).save(os.path.join(args.outputs, f'{image_name}_pred.png'))
        img.save(os.path.join(args.outputs,f'{image_name}_ori.png'))
        postprocess(comp_imgs[0]).save(os.path.join(os.path.join(args.outputs, 'images'), f'{image_name}.png'))
        postprocess(comp_imgs[0]).save(os.path.join(args.outputs, f'{image_name}.png'))
        print(f'saving to {os.path.join(args.outputs, image_name)}')


if __name__ == '__main__':
    main_worker(args)
