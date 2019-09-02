import os
import re
import sys
#sys.path.append('./')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
#import rtpose_vgg
#from rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender
#from lib.utils.paf_to_pose import find_peaks
from lib.pafprocess import pafprocess
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd_lr1.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='./ckpts/openpose.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

def draw_humans(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        # draw point
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return npimg
    
weight_name = './ckpts/openpose.pth'
model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name))
#model.cuda()
model.float()
model.eval()

if __name__ == "__main__":
    
    video_capture = cv2.VideoCapture('test.mp4')
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    f_out = cv2.VideoWriter('output.avi', fourcc, 30, (frame_width, frame_height))
    i = 0

    while True:
        # Capture frame-by-frame
        ret, oriImg = video_capture.read()
        
        shape_dst = np.min(oriImg.shape[0:2])

        # Get results of original image
        #multiplier = get_multiplier(oriImg)

        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')
            print(im_scale)
        
        '''          
        heatmap_peaks = np.zeros_like(heatmap)
        #for i in range(19):
        #    heatmap_peaks[:,:,i] = find_peaks(heatmap[:,:,i], oriImg)
        heatmap_peaks = heatmap_peaks.astype(np.float32)
        heatmap = heatmap.astype(np.float32)
        paf = paf.astype(np.float32)

        #C++ postprocessing      
        pafprocess.process_paf(heatmap_peaks, heatmap, paf)
        print("Mid")

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heatmap.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heatmap.shape[0],
                    pafprocess.get_part_score(c_idx)
                )

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)
        
        '''    
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        out = draw_humans(oriImg, humans)

        # Display the resulting frame
        print("End")
        f_out.write(out)
        cv2.imwrite('./results/frame_'+str(i)+'.png', out)
        i = i + 1
        cv2.imshow('Video', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
