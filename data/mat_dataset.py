import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
# from PIL import Image
# import PIL
from pdb import set_trace as st

import scipy.io as sio
import random
import numpy as np
from PIL import Image

class MatDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)

        self.data_paths = make_dataset(self.dir)
        self.size = len(self.data_paths)
        # print('!!! data set size %d\n, dir %s'%(self.size, self.dir))

        self.fineSize = opt.fineSize
        self.osize = opt.loadSize
        self.transform = get_transform(opt)
        self.phase = opt.phase

    def __getitem__(self, index):
        dataPath = self.data_paths[index % self.size]

        data = sio.loadmat(dataPath)
        data = data['data']
        rgb = data['rgb'][0,0]
        depth = data['depth'][0,0]

        # crop image to fineSize(256 for default)
        offset = 0
        if self.phase == 'train':
            offset = random.randint(0, self.osize-self.fineSize)
        else:
            offset = int(np.floor((self.osize - self.fineSize)/2))

        rgb_crop = rgb[offset:offset+self.fineSize, offset:offset+self.fineSize, :]
        depth_crop = depth[offset:offset+self.fineSize, offset:offset+self.fineSize]

        # fill depth values in all channels
        depth_final = np.ones((self.fineSize, self.fineSize, 3))
        depth_final[:,:,0] = depth_crop
        depth_final[:,:,1] = depth_crop
        depth_final[:,:,2] = depth_crop

        # rgb_final = rgb_crop.transpose((2, 0, 1))
        rgb_temp = Image.fromarray(rgb_crop.astype('uint8'),'RGB')
        rgb_final = self.transform(rgb_temp)
        depth_final = depth_final.transpose((2, 0, 1))

        # numpy.array(PIL.Image.open('xxx').convert('RGB')) can handle image directly

        return {'A': rgb_final, 'B': depth_final,
                'A_paths': dataPath, 'B_paths': dataPath}

    def __len__(self):
        return self.size

    def name(self):
        return 'MatDataset'
