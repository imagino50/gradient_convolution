from CannyEdge.utils import to_ndarray
from CannyEdge.core import (gs_filter, 
                            gradient_intensity, 
                            suppression, supress_non_max,
                            threshold,
                            convolve,
                            tracking)

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import argparse
from scipy import misc

# Argparse
parser = argparse.ArgumentParser(description='Educational Canny Edge Detector')
parser.add_argument('source', metavar='src', help='image source (jpg, png)')
parser.add_argument('sigma', type=float, metavar='sigma', help='Gaussian smoothing parameter')
parser.add_argument('t', type=int, metavar='t', help='lower threshold')
parser.add_argument('T', type=int, metavar='T', help='upper threshold')
parser.add_argument("--all", help="Plot all in-between steps")
args = parser.parse_args()

def ced(img_file, sigma, t, T, all=False):
    img = to_ndarray(img_file)
    if not all:
        # avoid copies, just do all steps:
        img = gs_filter(img, sigma)
        img, D = gradient_intensity(img)
        img = suppression(img, D)
        img, weak = threshold(img, t, T)
        img = tracking(img, weak)
        return [img]
    else:
        # make copies, step by step
        img1 = gs_filter(img, sigma)
        img2, D = gradient_intensity(img1)
        img3 = suppression(copy(img2), D)
        img4, weak, strong = threshold(copy(img3), t, T)
        img5 = tracking(copy(img4), weak)
        return [to_ndarray(img_file), img1, img2, D, img3, img4, img5]

def ced2(img_file, sigma, t, T, all=False):
    img = to_ndarray(img_file)
    if not all:
        # avoid copies, just do all steps:
        img = gs_filter(img, sigma)
        img, D = gradient_intensity(img)
        img = suppression(img, D)
        img, weak = threshold(img, t, T)
        img = tracking(img, weak)
        return [img]
    else:
        # make copies, step by step
        img1 = gs_filter(img, sigma)
        #img2, D = compute_gradient(img1) #NEW
        Gm, Gd = gradient_intensity(img1)
        GmNms = supress_non_max(Gm, Gd, 2, 1.0)#NEW
        #img3 = suppression(copy(img2), Gd)
        GmNmsThres, weak, strong = threshold(copy(GmNms), t, T)
        feat = convolve(GmNmsThres, Gd, stride=1, thres=weak)
        img6 = tracking(copy(GmNmsThres), weak)
        #return [to_ndarray(img_file), img1, Gm, Gd, GmNms, GmNmsThres, feat[0], feat[1], feat[2], feat[3], img6]
        return [to_ndarray(img_file), GmNmsThres, feat[0], feat[1], feat[2], feat[3], img6]

def plot(img_list, img_list2, safe=False):
    for d, img in enumerate(img_list):
        plt.subplot(3, len(img_list), d+1),
        plt.imshow(img, cmap='gray'),
        plt.xticks([]),
        plt.yticks([])
    for d, img in enumerate(img_list2[0:len(img_list)]):
        plt.subplot(3, len(img_list) , len(img_list) + d+1),
        plt.imshow(img, cmap='gray'),
        plt.xticks([]),
        plt.yticks([])
    for d, img in enumerate(img_list2[len(img_list):]):
        plt.subplot(3, len(img_list) , len(img_list)*2 + d+1),
        plt.imshow(img, cmap='gray'),
        plt.xticks([]),
        plt.yticks([])
    plt.show()


img_list = ced(args.source, args.sigma, args.t, args.T, all=args.all)
img_list2 = ced2(args.source, args.sigma, args.t, args.T, all=args.all)
plot(img_list, img_list2)

