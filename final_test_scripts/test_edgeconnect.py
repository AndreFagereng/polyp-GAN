import sys
sys.path.append('edge-connect')

import os
import scipy
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as T

from PIL import Image
from glob import glob
from easydict import EasyDict
from src.models import EdgeModel, InpaintingModel
from src.utils import imsave

from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb

_DEBUG = True

config = EasyDict()

config.PATH_TO_DATA = ""     # Keep empty..

config.OUTPUT_FOLDER = "edgeconnect_without_polyp_edge"  # (real, fake)

config.IMAGE_DIR = "data/finetune_testset/images"#"data/kvasir-seq-pretrain/images"

# Needs to be from same segmented image
config.MASK_DIR = "data/finetune_testset/masks"
config.EDGE_DIR = "data/finetune_testset/edges"


config.PATH = 'test_models_finetuned_testsetremoved/edgeconnect'  # Path to trained model
config.DEVICE = 'cuda'

config.MODE = 0
config.N_IMAMGES_TO_GENERATE = 800

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
    img = imread(path)          # Image
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
    #ip = sorted(glob(config.PATH_TO_DATA + '/images/*.jpg'))
    ip = glob(config.PATH_TO_DATA + '{}/*.jpg'.format(config.IMAGE_DIR))
    #np.random.shuffle(ip)
    ip = sorted(ip)
    mp = sorted(glob(config.PATH_TO_DATA + '{}/*.jpg'.format(config.MASK_DIR)))
    ep = sorted(glob(config.PATH_TO_DATA + '{}/*.jpg'.format(config.EDGE_DIR)))

    # Sanity test the filenames
    #for a, b, c in zip(ip, mp, ep):
    #    assert a.split('/')[-1] == b.split('/')[-1] == c.split('/')[-1]

    print('load_data: Found {} images'.format(len(ip)))
    
    return ip, mp, ep


inpaint_model = InpaintingModel(config).to(config.DEVICE)
inpaint_model.load()
inpaint_model.eval()

from scipy import ndimage

# Create folders
if not os.path.isdir(config.OUTPUT_FOLDER):
    os.makedirs(config.OUTPUT_FOLDER + '/GENERATED_IMAGES')
    os.makedirs(config.OUTPUT_FOLDER + '/GENERATED_IMAGES_MASKS')

ip, mp, ep = load_data(config)
ip = ip[:config.N_IMAMGES_TO_GENERATE]
mp = mp[:config.N_IMAMGES_TO_GENERATE]
ep = ep[:config.N_IMAMGES_TO_GENERATE]

for i, (img_p, msk_p, edg_p) in enumerate(zip(ip, mp, ep)):
    img_filename = img_p.split('/')[-1]
    filename = msk_p.split('/')[-1]

    img, img_gray = load_image(img_p)
    mask = load_mask(msk_p)
    edge = load_edge(edg_p)

    angle_in_degrees = 0#np.random.randint(0, 180)
    mask = ndimage.rotate(mask, angle_in_degrees, reshape=False)
    edge = ndimage.rotate(edge, angle_in_degrees, reshape=False)

    mask = rgb2gray(mask)

    img, mask, edge = cuda(*(to_tensor(img), to_tensor(mask), to_tensor(edge)))
    
    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)
    edge = edge.unsqueeze(0)

    # COPY OVER EDGES 
    # 1, 1, 256, 256 -> 256, 256
    edge = edge.squeeze(0).squeeze(0) 
    mask = mask.squeeze(0).squeeze(0)

    edge = (edge * (1 - (1 - mask)).float())
    edge = edge.unsqueeze(0).unsqueeze(0)

    save_edge = postprocess(edge)[0]
    save_mask = postprocess(mask.unsqueeze(0).unsqueeze(0))[0]
    #break

    image_edge = canny(img_gray, sigma=config.SIGMA, mask=None).astype(np.float)
    image_edge = to_tensor(image_edge).to(config.DEVICE)

    image_edge = image_edge.squeeze(0)
    image_edge_without = image_edge
    save_image_edge = postprocess(image_edge.unsqueeze(0).unsqueeze(0))[0]

    image_edge = (image_edge * (1 - mask).float())
    image_edge = image_edge.unsqueeze(0).unsqueeze(0)
    #image_edge = postprocess(image_edge)[0]
    
    image_edge = image_edge.squeeze(0).squeeze(0)

    love = image_edge.float() + edge.squeeze(0).squeeze(0).float()
    love = love.unsqueeze(0).unsqueeze(0)
    

    
    save_love = postprocess(love)[0]
    #imsave(love, config.OUTPUT_FOLDER + '/image_edge.jpg')
    #break    

    #edge = love
    edge = image_edge_without.unsqueeze(0).unsqueeze(0)
    print(edge.shape)

    outputs = inpaint_model(img, edge, mask) 
    outputs_merged = (outputs * mask) + (img * (1 - mask))

    output = postprocess(outputs_merged)[0]
        
    imsave(output, config.OUTPUT_FOLDER + '/GENERATED_IMAGES/{}'.format(img_filename))
    imsave(save_mask, config.OUTPUT_FOLDER + '/GENERATED_IMAGES_MASKS/{}'.format(img_filename))
    
    if _DEBUG:
        # Generated image -> image filename
        # Used mask -> Mask filename _ image_filename
        image_name = os.path.join(config.OUTPUT_FOLDER, '{}_{}'.format(i,img_filename))
        mask_name = os.path.join(config.OUTPUT_FOLDER, '{}_mask_{}'.format(i, filename))
        edge_name = os.path.join(config.OUTPUT_FOLDER, '{}_polyp_edge_{}'.format(i, filename))
        edge_name_image = os.path.join(config.OUTPUT_FOLDER, '{}_image_edge_{}'.format(i, img_filename))
        used_edge_name = os.path.join(config.OUTPUT_FOLDER, '{}_merged_edges_{}'.format(i, img_filename))
        imsave(output, image_name)
        imsave(save_mask, mask_name)
        imsave(save_edge, edge_name)
        imsave(save_image_edge, edge_name_image)
        imsave(save_love, used_edge_name)



        # Save img, mask, edge

    # Save fake











