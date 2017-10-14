import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.output_nc = 1

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
# visualizer = Visualizer(opt)

dataset = iter(dataset)
test_indx = 9
for i in range(0,test_indx-1):
    dataset.next()
images = dataset.next()
# A_arr = images['A'].resize_(3,256,256).numpy()

model.set_input(images)
result = model.test()
# print(result)

import math
from PIL import Image
import numpy as np
# dpBais = math.log10(0.8);
# dpScale = math.log10(655) - dpBais;
# result_r = 10**(dpScale*result + dpBais)
# print(result_r)

visuals = model.get_current_visuals()
real_A = visuals['real_A']
fake_B = visuals['fake_B']
real_B = visuals['real_B']

realImg = Image.fromarray(real_A.astype('uint8'),'RGB')
realImg.show()

resultImg = Image.fromarray(fake_B,'RGB')
resultImg.show()


gtDepth = np.tile(real_B, (3, 1, 1))
# print(gtDepth)
gtdepthImg = Image.fromarray(gtDepth.transpose([1,2,0]),'RGB')
gtdepthImg.show()

# visuals = model.get_current_visuals()
# resultImg = visuals['fake_B']
# from PIL import Image
# resultImg=Image.fromarray(resultImg)
# resultImg.show()


# # create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# # test
# for i, data in enumerate(dataset):
#     if i >= opt.how_many:
#         break
#     model.set_input(data)
#     model.test()
#     visuals = model.get_current_visuals()
#     img_path = model.get_image_paths()
#     print('process image... %s' % img_path)
#     visualizer.save_images(webpage, visuals, img_path)

# webpage.save()
