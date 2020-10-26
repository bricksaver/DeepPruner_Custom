# ---------------------------------------------------------------------------
# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
#
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Shivam Duggal
# ---------------------------------------------------------------------------

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage.io
import numpy as np
import logging
from dataloader import kitti_submission_collector as ls
from dataloader import preprocess
from PIL import Image
from models.deeppruner import DeepPruner
from models.config import config as config_args
from setup_logging import setup_logging
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='DeepPruner')
parser.add_argument('--datapath', default='/',
                    help='datapath')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--save_dir', default='./',
                    help='save directory')
parser.add_argument('--logging_filename', default='./submission_kitti.log',
                    help='filename for logs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
torch.backends.cudnn.benchmark = True
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.cost_aggregator_scale = config_args.cost_aggregator_scale
args.downsample_scale = args.cost_aggregator_scale * 8.0

setup_logging(args.logging_filename)

if args.cuda:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
'''
print('ARGS.DATAPATH:',args.datapath) 
test_left_img, test_right_img = ls.datacollector(args.datapath)
print('test_left_img:', test_left_img)
print('test_right_img:', test_right_img)
'''
model = DeepPruner()
model = nn.DataParallel(model)

if args.cuda:
    model.cuda()

logging.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


if args.loadmodel is not None:
    logging.info("loading model...")
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=True)
    print('FINISHED LOAD MODEL')

def test(imgL, imgR):
    model.eval()
    with torch.no_grad():
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))

        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        refined_disparity = model(imgL, imgR)
        return refined_disparity


def main():

    name_save_idx = 0
    print('POINT 0.0')
    #for left_image_path, right_image_path in zip(test_left_img, test_right_img):
    #for left_image_path, right_image_path in zip(image_2, image_3):

    for i in range(2,3):
        
        name_save_idx += 1
        #left_img_path = './unity_chair/rgb_chair_left/'
        #right_img_path = './unity_chair/rgb_chair_right/'
        left_img_path = './data/unity_chair/rgb_chair_left/' + '{:04d}'.format(i) + '.jpg' #CHECK TO CHANGE B4 RUNNING
        print('i:', i)
        right_img_path = './data/unity_chair/rgb_chair_right/' + '{:04d}'.format(i) + '.jpg' #CHECK TO CHANGE B4 RUNNING
        #imgL = cv2.imread(image_L_path,0)
        #imgR = cv2.imread(image_R_path,0)
        imgL = cv2.imread(left_img_path)
        imgR = cv2.imread(right_img_path)
        
        print('POINT 0.1')
        #imgL = np.asarray(Image.open(left_image_path))
        #imgR = np.asarray(Image.open(right_image_path))
        print('imgL:',imgL)
        print('imgR:',imgR)
        print('POINT 0.2')
        #print('IMGL SHAPE:',imgL.shape)
        #iprint('IMGR SHAPE:',imgR.shape)
        processed = preprocess.get_transform()
        print('POINT 1')
        imgL = processed(imgL).numpy()
        imgR = processed(imgR).numpy()
        print('IMG 1',imgL)
        print('IMGR 1',imgR)
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])
        print('IMGL 2',imgL)
        print('IMGR 2',imgR)

        w = imgL.shape[3]
        h = imgL.shape[2]
        dw = int(args.downsample_scale - (w%args.downsample_scale + (w%args.downsample_scale==0)*args.downsample_scale))
        dh = int(args.downsample_scale - (h%args.downsample_scale + (h%args.downsample_scale==0)*args.downsample_scale))

        top_pad = dh
        left_pad = dw
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)

        disparity = test(imgL, imgR)
        print('POINT 0')
        print('DISPARITY:',disparity)
        print('DISPARITY TYPE:',type(disparity))
        print('DISPARITY SHAPE:',disparity.shape)
        #disparity = list(disparity)
        disparity = disparity.cpu().numpy()
        #disparity = np.array(disparity)
        
        print(disparity)
        print('DISPARITY SHAPE:',disparity.shape)
        print('TOP PAD:',top_pad)
        print('LEFT_PAD:',left_pad)
        #disparity = disparity[0, top_pad:, :-left_pad].data.cpu().numpy()  
        print('POINT 1')
        print('NEED TO FIX ISSUE BELOW')
        print('DISPARITY AFTER PAD:',disparity)

        print(args.save_dir)
        #print(left_img_path.split(('/')[-1]), (disparity * 256).astype('uint16'))
        print('DISPARITY AFTER SPLIT:',disparity)

        #skimage.io.imsave(os.path.join(args.save_dir, left_img_path.split(('/')[-1]), (disparity * 256).astype('uint16')))
        print('POINT 3')

        filename = 'unity_chair_estimated_disparity_pair_dp_' + '{:04d}'.format(name_save_idx) + '.png'
        #disparity = np.uint16(disparity)
        print('POINT 4')
        #img = Image.fromarray(disparity_comb, 'RGB')
        #i#img.save(filename)
        ('POINT 5')
        #plt.imsave('test.png',(disparity*256).astype('uint16'))
        cv2.imwrite(filename,disparity[0])

        #logging.info("Disparity for {} generated at: {}".format(left_image_path, os.path.join(args.save_dir, left_image_path.split('/')[-1])))


if __name__ == '__main__':
        main()
