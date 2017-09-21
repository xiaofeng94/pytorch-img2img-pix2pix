import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

import scipy.io as sio
# import random
import math
import numpy as np

class SingleMatDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.fineSize = opt.fineSize
        self.osize = opt.loadSize

        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        data = sio.loadmat(A_path)
        data = data['data']
        rgb = data['rgb'][0,0]
        depth = data['depth'][0,0]

        # crop image to fineSize(256 for default)
        offset = max(0, math.floor((self.osize - self.fineSize)/2)) # random.randint(0, self.osize-self.fineSize)
        rgb_crop = rgb[offset:offset+self.fineSize, offset:offset+self.fineSize, :]
        depth_crop = depth[offset:offset+self.fineSize, offset:offset+self.fineSize]

        # fill depth values in all channels
        depth_final = np.ones((self.fineSize, self.fineSize, 3))
        depth_final[:,:,0] = depth_crop
        depth_final[:,:,1] = depth_crop
        depth_final[:,:,2] = depth_crop

        rgb_fianl = rgb_crop.transpose((2, 0, 1))
        depth_final = depth_final.transpose((2, 0, 1))

        return {'A': rgb_fianl, 'B': depth_final,
                'A_paths': A_path, 'B_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
