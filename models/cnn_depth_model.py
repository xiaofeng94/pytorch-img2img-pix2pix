import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

import torch.nn as nn

class CNNDepthModel(BaseModel):
    def name(self):
        return 'CNNDepthModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchsz_inbatch, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchsz_inbatch, 1, 1, 1)

        # define my network
        self.net = networks.Conv5Fc4Net(opt.input_nc)

        assert(torch.cuda.is_available())
        if len(self.gpu_ids) > 0:
            self.net.cuda(device_id=self.gpu_ids[0])

        if not self.isTrain or opt.continue_train:
            self.load_model(self.net, 'cnn_depth', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterion = nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.schedulers = []

            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.schedulers.append(networks.get_scheduler(self.optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.net)
        print('-----------------------------------------------')
        self.parmas = list(self.net.parameters())
        print('len(self.parmas): {}'.format(len(self.parmas)))

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A[0].size()).copy_(input_A[0])
        self.input_B.resize_(input_B[0].size()).copy_(input_B[0])
        self.image_paths = input['A_paths']
        # only for viusal
        self.visual = input['visual']
        # self.mask = input['mask']


    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.depth_pred = self.net.forward(self.real_A)

    def backward(self):
        self.loss = self.criterion(self.depth_pred, self.real_B)
        self.loss.backward()
        print(self.depth_pred)
        print('---------')
        print(self.real_B)
        print('------------------------------')
        print(self.parmas[16].grad)


    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.depth_pred = self.net.forward(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.backward()        
        self.optimizer.step()

    def get_current_errors(self):
        print(self.loss.data.shape)
        return OrderedDict([('MSEloss', self.loss.data[0]),
                            ('2*MSEloss', self.loss.data[0]*2)])

    def get_current_visuals(self):
        # change self.depth_pred to image

        visual_A = self.visual[0].cpu().float().numpy().astype(np.uint8)
        # return OrderedDict([('real_A', real_A), ('visual_A', visual_A),('fake_B', fake_B), ('real_B', real_B)])
        return OrderedDict([('visual_A', visual_A)])

    def save(self, label):
        # self.save_network(self.netG, 'G', label, self.gpu_ids)
        # self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_model(self.net, 'cnn_depth', label, self.gpu_ids)
        
    def save_model(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])

    def load_model(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('save path: %s'%save_path)

        network.load_state_dict(torch.load(save_path))
