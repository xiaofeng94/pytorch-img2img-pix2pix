import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
# from util import html

import numpy as np

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

farplane = 5000

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
# visualizer = Visualizer(opt)


dpBais = np.log10(0.8);
dpScale = np.log10(farplane) - dpBais;

# measures
rel = 0 # average relative error
rms = 0 # root mean squared error
log10Err = 0 # average log10 error
thrAcc1 = 0 # accuracy with threshold
thrAcc2 = 0 # accuracy with threshold
thrAcc3 = 0 # accuracy with threshold
thrCount1 = 0
thrValue1 = 1.25
thrCount2 = 0
thrValue2 = 1.25**2
thrCount3 = 0
thrValue3 = 1.25**3
dataCount = 0

for i, data in enumerate(dataset):
    dataCount += 1
    depth_gt = data['B'].cpu().float().numpy()
    depth_gt = depth_gt[0,0,:,:]
    depth_real = 10**(dpScale*depth_gt + dpBais)

    model.set_input(data)
    result = model.test().transpose([1,2,0])
    pred_depth = np.sum(result,2)/3
    pred_depth_real = 10**(dpScale*pred_depth + dpBais)

    # if i == 0:
    #     print('depth_real:')
    #     print(depth_real.shape)
    #     print(depth_real)
    #     print('pred_depth_real:')
    #     print(pred_depth_real.shape)
    #     print(pred_depth_real)
    #     print('different:')
    #     print(pred_depth_real-depth_real)


    # compute measures
    elementNum = depth_real.size
    curr_rel = np.sum(np.abs(pred_depth_real-depth_real)/depth_real)/elementNum
    # temp = (pred_depth_real-depth_real)**2
    # print(temp.shape)
    # print(temp)
    curr_rms = np.sqrt( np.sum((pred_depth_real-depth_real)**2)/elementNum )
    curr_log10Err = np.sum(np.abs(np.log10(pred_depth_real)-np.log10(depth_real)))/elementNum

    rel = rel*(dataCount-1)/dataCount + curr_rel/dataCount
    rms = rms*(dataCount-1)/dataCount + curr_rms/dataCount
    log10Err = log10Err*(dataCount-1)/dataCount + curr_log10Err/dataCount                                                                                                                                                                         

    temp_array = np.zeros([depth_real.shape[0],depth_real.shape[1],2])
    temp_array[:,:,0] = depth_real/pred_depth_real
    temp_array[:,:,1] = pred_depth_real/depth_real
    max_array = np.max(temp_array,2)
    # print('3..')
    # print(depth_real.shape)

    for indx_x in np.arange(depth_real.shape[0]):                                                                                   
        for indx_y in np.arange(depth_real.shape[1]):
            if max_array[indx_x, indx_y] < thrValue3:
                thrCount3 += 1
            if max_array[indx_x, indx_y] < thrValue2:
                thrCount2 += 1
            if max_array[indx_x, indx_y] < thrValue1:
                thrCount1 += 1

    if dataCount%50 == 0:
        print ('{} images has been processed!'.format(dataCount))

print ('{} images has been processed!\n'.format(dataCount))

pointsNum = (dataCount*elementNum)
thrAcc1 = thrCount1/pointsNum
thrAcc2 = thrCount2/pointsNum
thrAcc3 = thrCount3/pointsNum

print ('rel: {}\nlog10Err: {}\nrms: {}\nthr1: {}\nthr2: {}\nthr3: {}'.format(rel,log10Err,rms,thrAcc1,thrAcc2,thrAcc3))
# dataset = iter(dataset)
# images = dataset.next()
# A_arr = images['A'].resize_(3,256,256).numpy()

# model.set_input_array(A_arr)
# result = model.test()
# print(result.shape)

# import math
# from PIL import Image
# dpBais = math.log10(0.8);
# dpScale = math.log10(655) - dpBais;
# result_r = 10**(dpScale*result + dpBais)
# print(result_r)

# resultImg = Image.fromarray(result_r)
# resultImg.show()

# visuals = model.get_current_visuals()
# resultImg = visuals['fake_B']
# from PIL import Image
# resultImg=Image.fromarray(resultImg)
# resultImg.show()

