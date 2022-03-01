import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

class ToTensor(object):
    def __call__(self, sample):
        entry = {}
        for k in sample:
            if k == 'rect':
                entry[k] = torch.IntTensor(sample[k])
            elif k == 'mask':
                if type(sample[k]) != type(None):
                    entry[k] = torch.FloatTensor(sample[k])
            else:
                entry[k] = torch.FloatTensor(sample[k])
        return entry
class KvasirDataset(Dataset):


    def __init__(self, root_path, args, transform=None, im_size=(256, 256), image_folder="", label_folder="", ) -> None:
        super().__init__()

        self.args = args

        if not image_folder:
            image_folder = "images"
        if not label_folder:
            label_folder = "progan_mask"
        
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.root_path = root_path

        self.filenames = os.listdir(os.path.join(
            self.root_path, self.image_folder
        ))

        self._label_path = os.path.join(self.root_path, self.label_folder)
        self._image_path = os.path.join(self.root_path, self.image_folder)

        self.im_size = im_size
        self.transform = transform
        print(self._label_path)
        self._mask_filenames = glob.glob(self._label_path + '/*.png') + glob.glob(self._label_path + '/*.jpg')


    def __len__(self):
        return len(self.filenames)
    
    
    def read_image_and_label(self, filename):
        image_file = os.path.join(self._image_path, filename)
        
        if self.args.mask_type == "custom":
            label_file = os.path.join(self._label_path, filename)
            label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
            (thresh, label) = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)
        elif self.args.mask_type == "custom_r":
            r_idx = np.random.randint(0, len(self._mask_filenames))
            label_file = self._mask_filenames[r_idx]
            label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
            (thresh, label) = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)

            
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        h, w, c = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            # Scale Image
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            h, w, _ = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1],:]
            im_scaled = np.transpose(im_scaled, [2, 0, 1])

            # Scale mask
            if  "custom" in self.args.mask_type:
                if label.shape == image.shape:
                    lb_scaled = cv2.resize(label, None, fx=ratio, fy=ratio)
                    lb_scaled = lb_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1]]
                    lb_scaled = lb_scaled.astype(np.float32)
                else:
                    lb_scaled = cv2.resize(label, self.im_size, interpolation=cv2.INTER_AREA)
            else:
                lb_scaled = None

        else:
            im_scaled = np.transpose(image, [2, 0, 1])


        return im_scaled, lb_scaled

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image, label = self.read_image_and_label(filename)
    
        sample = {'gt': image, 'mask': label} if type(label) != type(None) else {'gt': image}

        if self.transform:
            self.transform(sample)
        
        return sample





