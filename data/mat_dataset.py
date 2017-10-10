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

        transform_list = []
        transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        self.phase = opt.phase
        self.isFlip = opt.isTrain and not opt.no_flip

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
        if self.isFlip:
            flipFlag = random.randint(0, 1)
            if flipFlag < 1:
                rgb_crop = np.fliplr(rgb_crop)
                depth_crop = np.fliplr(depth_crop)

        depth_final = np.ones((1,self.fineSize, self.fineSize))
        depth_final[0] = np.abs(depth_crop)
        # fill depth values in all channels
        # depth_final = np.ones((self.fineSize, self.fineSize, 3))
        # depth_final[:,:,0] = depth_crop
        # depth_final[:,:,1] = depth_crop
        # depth_final[:,:,2] = depth_crop
        # depth_final = depth_final.transpose((2, 0, 1))

        rgb_final = np.abs(rgb_crop).transpose((2, 0, 1))
        # rgb_temp = Image.fromarray(rgb_crop.astype('uint8'),'RGB')
        # rgb_visual = self.transform(rgb_temp) # only for visual view
        rgb_visual = np.abs(rgb_crop)

        # numpy.array(PIL.Image.open('xxx').convert('RGB')) can handle image directly

        return {'A': rgb_final, 'B': depth_final, 'C':rgb_visual,
                'A_paths': dataPath, 'B_paths': dataPath}

    def __len__(self):
        return self.size

    def name(self):
        return 'MatDataset'
