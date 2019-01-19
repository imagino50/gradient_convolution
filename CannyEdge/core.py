""" Canny Edge Detection is based on the following five steps:

    1. Gaussian filter
    2. Gradient Intensity
    3. Non-maximum suppression
    4. Double threshold
    5. Edge tracking

    This module contains these five steps as five separate Python functions.
"""

# Module imports
from CannyEdge.utils import round_angle

# Third party imports
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
from scipy.ndimage import sobel, generic_gradient_magnitude, generic_filter
from skimage import filters as filters2

import numpy as np


def gs_filter(img, sigma):
    """ Step 1: Gaussian filter

    Args:
        img: Numpy ndarray of image
        sigma: Smoothing parameter

    Returns:
        Numpy ndarray of smoothed image
    """
    if type(img) != np.ndarray:
        raise TypeError('Input image must be of type ndarray.')
    else:
        return gaussian_filter(img, sigma)


def gradient_intensity(img):
    """ Step 2: Find gradients

    Args:
        img: Numpy ndarray of image to be processed (denoised image)

    Returns:
        G: gradient-intensed image
        D: gradient directions
    """

    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], 
         [-2, 0, 2], 
         [-1, 0, 1]], np.int32
    )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], 
         [0, 0, 0], 
         [-1, -2, -1]], np.int32
    )
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    # return the hypothenuse of (Ix, Iy)
    Gm = np.hypot(Ix, Iy)
    Gd = np.arctan2(Iy, Ix)

    return (Gm, Gd)

def suppression(img, Gd):
    """ Step 3: Non-maximum suppression

    Args:
        img: Numpy ndarray of image to be processed (gradient-intensed image)
        D: Numpy ndarray of gradient directions for each pixel in img

    Returns:
        ...
    """
    temp0 = 0
    temp90 = 0
    temp135 = 0
    temp45 = 0
    tempE = 0

    M, N = img.shape
    print("supress_non_max::h:%d,w:%d " % (M,N))
    Z = np.zeros((M,N), dtype=np.int32)

    Gd = np.rad2deg(Gd) % 180

    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(Gd[i, j])
            try:
                if where == 0:
                    temp0 += 1
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i,j] = img[i,j]
                elif where == 90:
                    temp90 += 1
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        Z[i,j] = img[i,j]
                elif where == 135:
                    temp135 += 1
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        Z[i,j] = img[i,j]
                elif where == 45:
                    temp45 += 1
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        Z[i,j] = img[i,j]
            except IndexError as e:
                """ Todo: Deal with pixels at the image boundaries. """
                tempE += 1
                pass
    print("suppression::temp45:%d,temp0:%d, temp135:%d,temp90:%d,tempE:%d" % (temp45,temp0,temp135,temp90,tempE))
    return Z

def supress_non_max(Gm, Gd, scan_dim, thres):
    """ Step 3: Non-maximum suppression
    Args:
        Gm (Numpy ndarray): Gradient-intensed image to be processed
        Gd (Numpy ndarray): Gradient directions for each pixel in img  in the range [0, +180]
    Returns:
        Gm_nms (Numpy ndarray): Non-maximum suppression from Gradient magnitude
    """
    temp0 = 0
    temp90 = 0
    temp135 = 0
    temp45 = 0

    Gm_nms = np.copy(Gm)
    h,w = Gm.shape
    print("supress_non_max::h:%d,w:%d" % (h,w))

    Gd = np.rad2deg(Gd) % 180

    #x-coordinates : vertical edges
    for x in range(scan_dim, h-scan_dim):
        #y-coordinates : horizontal edges
        for y in range(scan_dim, w-scan_dim):
            mag = Gm[x,y]
            #print("Gd[%d,%d]=%d" %(x,y,Gd[x,y]))
            #if mag < thres: continue
            
            if (Gd[x,y]<22.5) or (Gd[x,y]>=157.5): 
             dx, dy = 0, -1 #angle = 0
             temp0 += 1
            elif (Gd[x,y]>=22.5) and (Gd[x,y]<67.5): 
             dx, dy = -1, 1 #angle = 45
             temp45 += 1
            elif (Gd[x,y]>=67.5) and (Gd[x,y]<112.5): 
             dx, dy = -1, 0 #angle = 90
             temp90 += 1
            elif (Gd[x,y]>=112.5) and (Gd[x,y]<157.5): 
             dx, dy = -1, -1 #angle = 135
             temp135 += 1
            
            for i in range(1, scan_dim +1):
             #print("i %d" % (i))
             if (mag < Gm[x+dx*i,y+dy*i]) or (mag < Gm[x-dx*i,y-dy*i]):
              Gm_nms[x,y]=0
              #print("Gm_nms[%d,%d]=0 " % (x,y))
    print("supress_non_max::temp45:%d,temp0:%d, temp135:%d,temp90:%d" % (temp45,temp0,temp135,temp90))
    return Gm_nms


def threshold(img, t, T):
    """ Step 4: Thresholding
    Iterates through image pixels and marks them as WEAK and STRONG edge
    pixels based on the threshold values.

    Args:
        img: Numpy ndarray of image to be processed (suppressed image)
        t: lower threshold
        T: upper threshold

    Return:
        img: Thresholdes image

    """
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(50),
        'STRONG': np.int32(255),
    }

    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)

    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))

    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)

    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)

    return (img, cf.get('WEAK'), cf.get('STRONG'))

#TO DO : features map
def detecte_lines(Gm, Gd, scan_dim, thres, x, y):
    """Step 4: Detecte lines over the image following `Gradient Direction` and `Gradient Magnitude`.
    Args:
        param1 (Numpy ndarray): The Gradient Magnitude of the image.
        param2 (Numpy ndarray): The Gradient Direction of the image [0,+180].
        param3 (int): The dimension of the scan towards the up|down|right|left directions.
        param4 (int): The minimum value of the of the Gradient Magnitude.
        param5 (int): The x position from where to scan and detect lines.
        param6 (int): The y position from where to scan and detect lines.
    Returns:
        Numpy ndarray : The features map.
    """
    h,w = Gm.shape
    
    #x-coordinates (North to South): vertical edges
    #y-coordinates (West to East): horizontal edges
    #mag = Gm[x,y]
    
    isPattern = True

    vertical = False
    sum_v = 0
    moy_v = 0

    horizontal = False
    sum_h = 0
    moy_h = 0

    diag1 = False
    sum_d1 = 0
    moy_d1 = 0

    diag2 = False
    sum_d2 = 0
    moy_d2 = 0

    nb_items = scan_dim*2 +1

    rangeList = list(range(1, scan_dim +1)) + list(range(-1, -scan_dim -1))
    
    #print("Gd[%d,%d]=%d" %(x,y,Gd[x,y]))
    if (Gd[x,y]<22.5) or (Gd[x,y]>=157.5): #angle = 0
      dx, dy = -1, 0  # +90 & -90
      if((x + scan_dim < h) and (x - scan_dim >= 0)):
        for i in rangeList:
          if(Gm[x+dx*i,y+dy*i]<thres or (Gd[x+dx*i,y+dy*i]>=22.5 and Gd[x+dx*i,y+dy*i]<157.5)):
            isPattern = False
            #print("Gm[%d,%d]=%d : FALSE : Gd[%d,%d]=%d;Gm[%d,%d]=%d" %(x,y,Gm[x,y],x+dx*i,y+dy*i,Gd[x+dx*i,y+dy*i],x+dx*i,y+dy*i,Gm[x+dx*i,y+dy*i]))
            #sum_v += Gm[x+dx*i,y+dy*i] 
            break
      else:
        isPattern = False
      vertical = isPattern
    elif (Gd[x,y]>=22.5) and (Gd[x,y]<67.5): #angle = 45 
      dx, dy = -1, -1 # +135 & -45
      if((x + scan_dim < w) and (y - scan_dim >= 0) and (x - scan_dim >= 0) and (y + scan_dim < h)):
        for i in rangeList:
          if(Gm[x+dx*i,y+dy*i]<thres or (Gd[x+dx*i,y+dy*i]<22.5 and Gd[x+dx*i,y+dy*i]>=67.5)):
            isPattern = False
            #sum_d1 += Gm[x+dx*i,y+dy*i]
            break
      else:
        isPattern = False
      diag1 = isPattern
    elif (Gd[x,y]>=67.5) and (Gd[x,y]<112.5): #angle = 90
      dx, dy = 0, -1 # +180 & 0 
      if(y + scan_dim < w) and (y - scan_dim >= 0):
        for i in rangeList:
          if(Gm[x+dx*i,y+dy*i]<thres or (Gd[x+dx*i,y+dy*i]>67.5 and Gd[x+dx*i,y+dy*i]>=112.5)):
            isPattern = False
            #sum_h += Gm[x+dx*i,y+dy*i]
            break
      else:
        isPattern = False
      horizontal = isPattern
    elif (Gd[x,y]>=112.5) and (Gd[x,y]<157.5): #angle = 135
      dx, dy = -1, 1 # +45 & -135
      if((x + scan_dim < w) and (y - scan_dim >= 0) and (x - scan_dim >= 0) and (y + scan_dim < h)):
       for i in rangeList:
        if(Gm[x+dx*i,y+dy*i]<thres or (Gd[x+dx*i,y+dy*i]<112.5 and Gd[x+dx*i,y+dy*i]>=157.5)):
         isPattern = False
         #sum_d2 += Gm[x+dx*i,y+dy*i]
         break
      else:
        isPattern = False
      diag2 = isPattern
    
    if vertical == True:
      moy_v = thres #sum_v / nb_items
      print("x=%d,y=%d : vertical == True" %(x,y))

    if diag1 == True:
      moy_d1 = thres #sum_d1 / nb_items
      print("x=%d,y=%d : diag1 == True" %(x,y))

    if horizontal == True:
      moy_h = thres #sum_h / nb_items
      print("x=%d,y=%d : horizontal == True" %(x,y))

    if diag2 == True:
      moy_d2 = thres #sum_d2 / nb_items
      print("x=%d,y=%d : diag2 == True" %(x,y))
     
    return moy_v, moy_d1, moy_h, moy_d2

def convolve(Gm, Gd, stride, thres):
    """Step 5: Confolve `Gradient filter` over `image` using `stride`

    Args:
        param1 (int): The image to convolve.
        param2 (int): The Gradient Magnitude of the image.
        param3 (int): The Gradient Direction of the image [0,+180]
        param4 (int): The stride that is parameter of the convolution.
        param5 (int): The minimum value of the of the Gradient Magnitude.

    Returns:
        Numpy ndarray: The features map.
    """
    nb_filter = 10

    Gd = np.rad2deg(Gd) % 180
        
    # Get filter dimensions (Square)
    filter_dim = 5
    scan_dim = 2
    
    # Get image dimensions (Square)
    #nb_chan_img, img_dim, _ = image.shape 
    img_dim, _ = Gm.shape 
    print("img_dim: %d." % (img_dim))
	
    # Calculate output dimensions
    feature_dim = int((img_dim - filter_dim)/stride)+1 
    print("feature_dim: %d." % (feature_dim))
	
    # Initialize an empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros((nb_filter,feature_dim,feature_dim))
    
    # Convolve the image by the filter(s) over every part of the image, adding the bias at each step. 
    #for curr_filter in range(nb_filter):
    curr_y = scan_dim
    featMap_y = 0
    while curr_y + scan_dim < img_dim:
        curr_x = scan_dim
        featMap_x = 0
        while curr_x + scan_dim < img_dim:
            if (Gm[curr_x,curr_y] >= thres):
             #print("Gd[%d,%d]=%d" %(curr_x,curr_x,Gd[curr_x,curr_x]))
             #print("curr_x: %d. curr_y: %d. featMap_y: %d. featMap_x: %d." % (curr_x, curr_y, featMap_y, featMap_x ))
             #print("LOOP : Gm[%d,%d]=%d" %(curr_x,curr_y,Gm[curr_x,curr_y]))
             moy_v, moy_d1, moy_h, moy_d2 = detecte_lines(Gm,Gd,scan_dim,thres,curr_x,curr_y)
             feature_maps[0, featMap_x, featMap_y] = moy_v
             feature_maps[1, featMap_x, featMap_y] = moy_d1
             feature_maps[2, featMap_x, featMap_y] = moy_h
             feature_maps[3, featMap_x, featMap_y] = moy_d2
            curr_x += stride
            featMap_x += 1
        curr_y += stride
        featMap_y += 1

    # Return all feature maps.
    return feature_maps

def tracking(img, weak, strong=255):
    """ Step 5:
    Checks if edges marked as weak are connected to strong edges.

    Note that there are better methods (blob analysis) to do this,
    but they are more difficult to understand. This just checks neighbour
    edges.

    Also note that for perfomance reasons you wouldn't do this kind of tracking
    in a seperate loop, you would do it in the loop of the tresholding process.
    Since this is an **educational** implementation ment to generate plots
    to help people understand the major steps of the Canny Edge algorithm,
    we exceptionally don't care about perfomance here.

    Args:
        img: Numpy ndarray of image to be processed (thresholded image)
        weak: Value that was used to mark a weak edge in Step 4

    Returns:
        final Canny Edge image.
    """

    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                         or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                         or (img[i+1, j + 1] == strong) or (img[i-1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img
