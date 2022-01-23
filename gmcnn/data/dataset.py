import os
import cv2
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
                entry[k] = torch.FloatTensor(sample[k])
            else:
                entry[k] = torch.FloatTensor(sample[k])
        return entry
class KvasirDataset(Dataset):


    def __init__(self, root_path, transform=None, im_size=(256, 256), image_folder="", label_folder="") -> None:
        super().__init__()

        if not image_folder:
            image_folder = "images"
        if not label_folder:
            label_folder = "masks"
        
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
    def __len__(self):
        return len(self.filenames)
    
    def read_image_and_label(self, filename):
        image_file = os.path.join(self._image_path, filename)
        label_file = os.path.join(self._label_path, filename)

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        (thresh, label) = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)

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
            lb_scaled = cv2.resize(label, None, fx=ratio, fy=ratio)
            lb_scaled = lb_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1]]
            
        else:
            im_scaled = np.transpose(image, [2, 0, 1])
        #print(lb_scaled)
        #zero_indices = lb_scaled == 0
        #ones_indices = lb_scaled == 1
        #lb_scaled[zero_indices] = 1
        #lb_scaled[ones_indices] = 0
        #print('sep')
        #print(lb_scaled)

        return im_scaled, lb_scaled.astype(np.float32)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image, label = self.read_image_and_label(filename)
        sample = {'gt': image, 'mask': label}

        if self.transform:
            self.transform(sample)
        
        return sample





