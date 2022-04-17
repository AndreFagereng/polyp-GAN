
import os
import numpy as np
import scipy
import torchvision.transforms.functional as F
from PIL import Image
from glob import glob
from skimage.feature import canny
from easydict import EasyDict
from skimage.color import rgb2gray, gray2rgb
from scipy.misc import imread

config = EasyDict()

config.SIGMA = 2
config.INPUT_SIZE = 256
config.PATH_TO_DATA = "data/kvasir-seq-1000/Kvasir-SEG"

config.IMAGE_DIR = "images"
config.MASK_DIR = "masks"
config.OUTPUT_FOLDER = "edges"

def postprocess(img):
    # [0, 1] => [0, 255]
    img = (1 - img)
    img = img * 255.0
    
    #img = img.permute(0, 2, 3, 1)'
    return img.int()

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t

def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    print('Saving', im.size)
    im.save(path)


def resize(img, height=config.INPUT_SIZE, width=config.INPUT_SIZE):
    imgh, imgw = img.shape[0:2]
    img = scipy.misc.imresize(img, [height, width])
    return img

def load_data(config):
    ip = sorted(glob(config.PATH_TO_DATA + '/{}/*.jpg'.format(config.IMAGE_DIR)))
    mp = sorted(glob(config.PATH_TO_DATA + '/{}/*.jpg'.format(config.MASK_DIR)))

    # Sanity test the filenames
    for a, b in zip(ip, mp):
        assert a.split('/')[-1] == b.split('/')[-1]

    print('load_data: Found {} images'.format(len(ip)))
    
    return ip, mp

def load_mask(path):
    mask = imread(path)
    mask = resize(mask)
    mask = (mask > 0).astype(np.uint8) * 255 # threshold due to interpolation
    return mask

def load_image(path):
    img = imread(path)          # Image
    img = resize(img)
    img_gray = rgb2gray(img)     # Image grayscale (dont need)
    return img, img_gray

edge_folder = os.path.join(config.PATH_TO_DATA, config.OUTPUT_FOLDER)
if not os.path.isdir(edge_folder):
    os.makedirs(edge_folder)
    
ip, mp = load_data(config)

for (img_path, mask_path) in zip(ip, mp):
    

    filename = img_path.split('/')[-1]

    img, img_gray = load_image(img_path)
    mask = load_mask(mask_path)
    #mask = rgb2gray(mask)

    edge = canny(img_gray, sigma=config.SIGMA, mask=None).astype(np.float)


    edge = to_tensor(edge)
    edge = postprocess(1 - edge)[0]
    
    imsave(edge, config.PATH_TO_DATA + '/{}/{}'.format(config.OUTPUT_FOLDER, filename))
    #imsave(mask, config.OUTPUT_FOLDER + '/{}/{}'.format('masks', filename))




