from skimage.feature import canny
from PIL import Image
import numpy as np
from scipy.misc import imread
import numpy as np
import torchvision.transforms.functional as F
#from imageio import imread
#from skimage.transform import resize
import random
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
#from .utils import create_mask
def imsave(img, path):
    print(img.shape)
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    print('Saveing', im.size)
    im.save(path)

def postprocess(img):
    # [0, 1] => [0, 255]
    img = (1 - img)
    img = img * 255.0
    
    #img = img.permute(0, 2, 3, 1)
    return img.int()

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t

def resize(img, height, width, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    img = scipy.misc.imresize(img, [height, width])
    #img = resize(img, [height, width])
    return img

path_to_img = "PLS/images"
path_to_mask = "PLS/masks"
path_to_edges = "PLS/edges"

#image_path = "data/test-set/bbps-2-3/0d1797b1-db98-41e4-9251-8ae3f832e4aa.jpg"
filename = "ck2bxknhjvs1x0794iogrq49k.jpg"

image_path = "data/finetune/images_test/{}".format(filename)
mask_path = "data/finetune/masks_test/{}".format(filename)

img = imread(image_path)
img = rgb2gray(img)

im = Image.fromarray(img.astype(np.uint8))
im.save('test.jpg')

sigma = 1#random.randint(1, 4)
print('SIGMA', sigma)
mask=None

edges = canny(img, sigma=sigma, mask=mask).astype(np.float)

edges = to_tensor(edges)

edges = postprocess(1 - edges)[0]
print(edges)

import shutil


#test_img = "data/test-set/pretraining_test_set/96ab9e93-1678-4736-bda8-a2bfe2599492.jpg"
test_img = "data/kvasir-seq-pretrain/images/0a0d358f-7389-4e76-b4e3-8139004d406a.jpg"
#test_img = "data/kvasir-seq-pretrain/images/aaf7233d-0a03-47aa-8caf-cd83a51bdb6f.jpg"
#test_img = "data/kvasir-seq-pretrain/images/aaeff39b-5321-4229-96b7-60cbd26d61bd.jpg"
#test_img = "data/kvasir-seq-pretrain/images/aaccf08e-0fdb-4972-9fe2-830b1e1379a5.jpg"
#test_img = "data/kvasir-seq-pretrain/images/aad36a28-bf6a-4590-a9d7-087e797aa224.jpg"
#test_img = "data/kvasir-seq-pretrain/images/aafb0683-7f8b-4243-b460-6e7c65f6449b.jpg"
#test_img = "data/kvasir-seq-pretrain/images/aad69695-a1aa-4c41-8068-8b6e4beb0151.jpg"
#test_img = "data/finetune/images_test/cju1aqqv02qwz0878a5cyhr67.jpg"
#test_img = "data/finetune/images_test/cju7b3f5h1sm40755i572jden.jpg"
test_img = "data/test-set/pretraining_test_set/96ff1f9f-91dd-4598-97a5-5416eb0b3c03.jpg"

shutil.copy(test_img, path_to_img + '/ck2bxknhjvs1x0794iogrq49k.jpg')
#shutil.copy(image_path, path_to_img + '/{}'.format(filename))
shutil.copy(mask_path, path_to_mask + '/{}'.format(filename))
imsave(edges, path_to_edges + '/{}'.format(filename))

