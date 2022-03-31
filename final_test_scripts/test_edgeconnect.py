import sys
sys.path.append('edge-connect')

import os
import scipy
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image
from glob import glob
from easydict import EasyDict
from src.models import EdgeModel, InpaintingModel
from src.utils import imsave

from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb

_DEBUG = False

config = EasyDict()

config.PATH_TO_DATA = "PLS"     # (images, masks, edges)
config.OUTPUT_FOLDER = "folder_to_output"  # (real, fake)

config.PATH = 'test_models_finetuned_testsetremoved/edgeconnect'  # Path to trained model
config.DEVICE = 'cuda'

config.MODE = 0

# Model 
config.GAN_LOSS = "nsgan"             # nsgan | lsgan | hinge
config.GPU = [0] 
config.LR = 0.0001                    # learning rate
config.D2G_LR = 0.1                   # discriminator/generator learning rate ratio
config.BETA1 =  0.0                   # adam optimizer beta1
config.BETA2 = 0.9                    # adam optimizer beta2
config.BATCH_SIZE = 1                 # input batch size for training
config.INPUT_SIZE = 256               # input image size for training 0 for original size
config.SIGMA = 2                      # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
config.MAX_ITERS = 2e6                # maximum number of iterations to train the model

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t

def cuda(*args):
    return (item.to(config.DEVICE) for item in args)

def resize(img, height=config.INPUT_SIZE, width=config.INPUT_SIZE):
    imgh, imgw = img.shape[0:2]
    img = scipy.misc.imresize(img, [height, width])
    return img

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()


def load_image(path):
    img = imread(img_p)          # Image
    img = resize(img)
    img_gray = rgb2gray(img)     # Image grayscale (dont need)
    return img, img_gray

def load_mask(path):
    mask = imread(path)
    mask = resize(mask)
    mask = (mask > 0).astype(np.uint8) * 255 # threshold due to interpolation
    return mask

def load_edge(path):
    edge = imread(path)
    edge = resize(edge)
    return edge

def load_data(config):
    ip = sorted(glob(config.PATH_TO_DATA + '/images/*.jpg'))
    mp = sorted(glob(config.PATH_TO_DATA + '/masks/*.jpg'))
    ep = sorted(glob(config.PATH_TO_DATA + '/edges/*.jpg'))

    # Sanity test the filenames
    for a, b, c in zip(ip, mp, ep):
        assert a.split('/')[-1] == b.split('/')[-1] == c.split('/')[-1]

    print('load_data: Found {} images'.format(len(ip)))
    
    return ip, mp, ep


inpaint_model = InpaintingModel(config).to(config.DEVICE)
inpaint_model.load()
inpaint_model.eval()

# Create folders
if not os.path.isdir(config.OUTPUT_FOLDER):
    os.makedirs(config.OUTPUT_FOLDER)

ip, mp, ep = load_data(config)
for (img_p, msk_p, edg_p) in zip(ip, mp, ep):

    img, img_gray = load_image(img_p)
    mask = load_mask(msk_p)
    edge = load_edge(edg_p)

    img, img_gray, mask, edge = cuda(*(to_tensor(img), to_tensor(img_gray), to_tensor(mask), to_tensor(edge)))
    
    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)
    edge = edge.unsqueeze(0)
    print(img.shape)

    outputs = inpaint_model(img, edge, mask) 
    outputs_merged = (outputs * mask) + (img * (1 - mask))

    output = postprocess(outputs_merged)[0]
    
    imsave(output, config.OUTPUT_FOLDER + '/test.jpg')

    if _DEBUG:
        pass 
        # Save img, mask, edge

    # Save fake











