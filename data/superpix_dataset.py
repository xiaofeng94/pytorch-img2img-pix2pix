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

class SuperPixDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)

        self.data_paths = make_dataset(self.dir)
        self.size = len(self.data_paths)
        # print('!!! data set size %d\n, dir %s'%(self.size, self.dir))
        # self.fineSize = opt.fineSize
        # self.osize = opt.loadSize

        self.phase = opt.phase
        self.isTrain = opt.isTrain
        # self.isFlip = opt.isTrain and not opt.no_flip

        self.spNum = 950 # training super pixel number per image, redundancy exists
        self.patchSize = opt.batchsz_inbatch
        self.patchNum = int(950/self.patchSize) # patch number per image

    def __getitem__(self, index):
        indx = index%(self.size*self.patchNum)
        image_indx = int(np.floor(indx/self.patchNum))
        indx_offset = int(np.floor(indx%self.patchNum))

        dataPath = self.data_paths[image_indx]

        try:
            data = sio.loadmat(dataPath)
        except Exception as e:
            print('exception occur to load {}, No. {}'.format(dataPath, image_indx))
        else:
            data = data['data'][0,0]
            # print(type(data))
            imageData = data['imageData'] #image with padding filled
            centroids = data['centroids']
            depthData = data['depthData']
            patchSize = data['patchSize']
            visual = data['visual']
            depthMap = data['depthMap']
            mask = data['mask']
            sp_num = data['sp_num']

            # print('index: {}'.format(image_indx))
            # print(patchSize.shape)

            halfPatchSize = [int(patchSize[0,0]/2), int(patchSize[0,1]/2)]
            rgb_final = np.ones([self.patchSize, 3, patchSize[0,0], patchSize[0,1]])
            depth_final = np.ones([self.patchSize,1])

            # print('final shape')
            # print(rgb_final.shape)

            patch_start = indx_offset*self.patchSize
            for i in np.arange(self.patchSize):
                centerX = centroids[patch_start+i,0]
                centerY = centroids[patch_start+i,1]
                curr_patch = imageData[centerY-halfPatchSize[1]:centerY+halfPatchSize[1], 
                                        centerX-halfPatchSize[0]:centerX+halfPatchSize[0],:]

                rgb_final[i,:,:,:] = curr_patch.transpose([2,0,1])

                depth_final[i,0] = depthData[patch_start+i,0]

            rgb_visual = visual*255

            return {'A': rgb_final, 'B': depth_final, 'visual':rgb_visual,
                    'A_paths': dataPath}
        finally:
            pass

    def __len__(self):
        return int(self.size*self.patchNum)

    def name(self):
        return 'SuperPixDataset'
