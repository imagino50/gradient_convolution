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
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
    )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
    )
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    # return the hypothenuse of (Ix, Iy)
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return (G, D)

def compute_gradient(image):
    """ Step 2: Find gradients
    Args:
        image (Numpy ndarray): Image to be processed (denoised image)
    Returns:
        Gm (Numpy ndarray): Gradient-intensed image
        Gd (Numpy ndarray): Gradient directions
    """
    #Find the horizontal edges (x-coordinates)of an image using the Sobel transform. 
    sob_h = filters2.sobel_h(image, mask=None)

    #Find the vertical edges (y-coordinates) of an image using the Sobel transform. 
    sob_v = filters2.sobel_v(image, mask=None)

    ##Find the edge magnitude of an image using the Sobel transform. """ 
    gradient_mag = np.sqrt(sob_h ** 2 + sob_v ** 2)
    gradient_mag /= np.sqrt(2)
        
    ##Find the edge angle of an image using the Sobel transform. """ 
    #arctan2(y-coordinates,x-coordinates)
    #Return an array of angles in radians in the range [-pi, pi]. 
    angleRad = np.arctan2(sob_h, sob_v) # Counterclockwise 

    #Convert angles from radians to degrees. Input angle must be in [0,180]
    gradient_dir = np.rad2deg(angleRad) % 180

    #Convert an array of angles in degrees in the range [0, +180]. 
    #gradient_dir[gradient_dir<0] += 180

    return gradient_mag, gradient_dir

def suppression(img, D):
    """ Step 3: Non-maximum suppression

    Args:
        img: Numpy ndarray of image to be processed (gradient-intensed image)
        D: Numpy ndarray of gradient directions for each pixel in img

    Returns:
        ...
    """

    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)

    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(D[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i,j] = img[i,j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        Z[i,j] = img[i,j]
                elif where == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        Z[i,j] = img[i,j]
                elif where == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        Z[i,j] = img[i,j]
            except IndexError as e:
                """ Todo: Deal with pixels at the image boundaries. """
                pass
    return Z

def supress_non_max(Gm, Gd, scan_dim, thres):
    """ Step 3: Non-maximum suppression
    Args:
        Gm (Numpy ndarray): Gradient-intensed image to be processed
        Gd (Numpy ndarray): Gradient directions for each pixel in img  in the range [0, +180]
    Returns:
        Gm_nms (Numpy ndarray): Non-maximum suppression from Gradient magnitude
    """
    Gm_nms = np.copy(Gm)
    h,w = Gm.shape
    #x-coordinates : horizontal edges
    for x in range(scan_dim, w-scan_dim):
        #y-coordinates : vertical edges
        for y in range(scan_dim, h-scan_dim):
            mag = Gm[y,x]
            #if mag < thres: continue

            #print("Gm[%d,%d]=%f " % (y,x,mag))
            #angle = 0
            if (Gd[y,x]<=22.5 or Gd[y,x]>157.5): dy, dx = 0, -1
            
            #angle = 45
            if (Gd[y,x]>22.5 and Gd[y,x]<=67.5): dy, dx = -1, 1
        
            #angle = 90
            if (Gd[y,x]>67.5 and Gd[y,x]<=112.5): dy, dx = -1, 0
                
            #angle = 135
            if (Gd[y,x]>112.5 and Gd[y,x]<=157.5): dy, dx = -1, -1
            
            for i in range(1, scan_dim +1):
             #print("i %d" % (i))
             if mag <= Gm[y+dy*i,x+dx*i] and mag <= Gm[y-dy*i,x-dx*i]: 
              Gm_nms[y,x]=0
              #print("Gm_nms[%d,%d]=0 " % (x,y))

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

    return (img, cf.get('WEAK'))

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
