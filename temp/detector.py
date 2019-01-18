from CannyEdge.utils import to_ndarray
from CannyEdge.core import (gs_filter, 
                            gradient_intensity, compute_gradient, 
                            suppression, supress_non_max,
                            threshold, tracking)

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
        img4, weak = threshold(copy(img3), t, T)
        img5 = tracking(copy(img4), weak)
        return [to_ndarray(img_file), img1, img2, img3, img4, img5]

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
        img2, D = compute_gradient(img1) #OLD
        img3 = supress_non_max(img2, D, 1, 1.0)#OLD
        img4, weak = threshold(copy(img3), t, T)
        img5 = tracking(copy(img4), weak)
        return [to_ndarray(img_file), img1, img2, img3, img4, img5]

def plot(img_list, img_list2, safe=False):
    for d, img in enumerate(img_list):
        plt.subplot(2, len(img_list), d+1),
        plt.imshow(img, cmap='gray'),
        plt.xticks([]),
        plt.yticks([])
    for d, img in enumerate(img_list2):
        plt.subplot(2, len(img_list) , len(img_list) + d+1),
        plt.imshow(img, cmap='gray'),
        plt.xticks([]),
        plt.yticks([])
    plt.show()


img_list = ced(args.source, args.sigma, args.t, args.T, all=args.all)
img_list2 = ced2(args.source, args.sigma, args.t, args.T, all=args.all)
plot(img_list, img_list2)

