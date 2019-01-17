""" Edge Detection is based on the following five steps:
    1. Gaussian filter
    2. Gradient Intensity
    3. Non-maximum suppression
    4. Double threshold
    5. Edge tracking
    This module contains these five steps as five separate Python functions.
"""
from skimage import data 
from skimage import filters 
import numpy as np

### TODO : Padding + cross edge detection

def compute_gradient(image):
    """ Step 2: Find gradients
    Args:
        image (Numpy ndarray): Image to be processed (denoised image)
    Returns:
        Gm (Numpy ndarray): Gradient-intensed image
        Gd (Numpy ndarray): Gradient directions
    """
    #Find the horizontal edges (x-coordinates)of an image using the Sobel transform. 
    sob_h = filters.sobel_h(image, mask=None)

    #Find the vertical edges (y-coordinates) of an image using the Sobel transform. 
    sob_v = filters.sobel_v(image, mask=None)

    ##Find the edge magnitude of an image using the Sobel transform. """ 
    gradient_mag = np.sqrt(sob_h ** 2 + sob_v ** 2)
    gradient_mag /= np.sqrt(2)
        
    ##Find the edge angle of an image using the Sobel transform. """ 
    #arctan2(y-coordinates,x-coordinates)
    #Return an array of angles in radians in the range [-pi, pi]. 
    angleRad = np.arctan2(sob_v, sob_h) # Counterclockwise 

    #Convert angles from radians to degrees.
    gradient_dir = np.rad2deg(angleRad)

    #Convert an array of angles in degrees in the range [0, +180]. 
    gradient_dir[gradient_dir<0] += 180

    return gradient_mag, gradient_dir

#https://github.com/fubel/PyCannyEdge/blob/master/CannyEdge/core.py
def supress_non_max(Gm, Gd, scan_dim=2, thres=1.0):
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
    for x in range(1, w-1):
        #y-coordinates : vertical edges
        for y in range(1, h-1):
            #angle = 0
            if (Gd[y,x]<=22.5 or Gd[y,x]>157.5): dy, dx = 0, -1
            
            #angle = 45
            if (Gd[y,x]>22.5 and Gd[y,x]<=67.5): dy, dx = -1, 1
        
            #angle = 90
            if (Gd[y,x]>67.5 and Gd[y,x]<=112.5): dy, dx = -1, 0
                
            #angle = 135
            if (Gd[y,x]>112.5 and Gd[y,x]<=157.5): dy, dx = -1, -1
            
            for i in range(1, scan_dim):
                if Gm[y,x] <= Gm[y+dy*i,x+dx*i] and Gm[y,x] <= Gm[y-dy*i,x-dx*i]: Gm_nms[y,x]=0

    return Gm_nms

#TO DO : features map
def detecte_lines(Gm, Gd, scan_dim, thres, y, x):
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
    
    #y-coordinates (North to South): vertical edges
    #x-coordinates (West to East): horizontal edges
    mag = Gm[y,x]
    
    vertical = True
    sum_v = 0
    moy_v = 0

    horizontal = True
    sum_h = 0
    moy_h = 0

    diag1 = True
    sum_d1 = 0
    moy_d1 = 0

    diag2 = True
    sum_d2 = 0
    moy_d2 = 0

    nb_items = scan_dim*2 +1

    rangeList = list(range(1, scan_dim)) + list(range(-1, -scan_dim))
    
    #angle = 0 
    if (Gd[y,x]<=22.5 or Gd[y,x]>157.5):
      dy, dx = -1, 0 # +90 & -90
      if((y + scan_dim < h) and (y - scan_dim >= 0)):
        for i in rangeList:
          if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]>22.5 and Gd[y+dy*i,x+dx*i]<=157.5)):
            vertical = False
            break
        sum_v += Gm[y+dy*i,x+dx*i]
      else:
       vertical = False  
       
    #angle = 45 
    if (Gd[y,x]>22.5 and Gd[y,x]<=67.5):
      dy, dx = -1, -1 # +135 & -45
      if((x + scan_dim < w) and (y - scan_dim >= 0) and (x - scan_dim >= 0) and (y + scan_dim < h)):
       for i in rangeList:
        if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]<=22.5 and Gd[y+dy*i,x+dx*i]>67.5)):
         diag1 = False
         break
        sum_d1 += Gm[y+dy*i,x+dx*i]
      else:
       diag1 = False  
       
    #angle = 90
    if (Gd[y,x]>67.5 and Gd[y,x]<=112.5):
      dy, dx = 0, -1 # +180 & 0 
      if(x + scan_dim < w) and (x - scan_dim >= 0):
       for i in rangeList:
        if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]>=67.5 and Gd[y+dy*i,x+dx*i]>112.5)):
         horizontal = False
         break
        sum_h += Gm[y+dy*i,x+dx*i]
      else: 
       horizontal = False
     
    #angle = 135
    if (Gd[y,x]>112.5 and Gd[y,x]<=157.5):
      dy, dx = -1, 1 # +45 & -135
      if((x + scan_dim < w) and (y - scan_dim >= 0) and (x - scan_dim >= 0) and (y + scan_dim < h)):
       for i in rangeList:
        if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]<=112.5 and Gd[y+dy*i,x+dx*i]>157.5)):
         diag2 = False
         break
        sum_d2 += Gm[y+dy*i,x+dx*i]
      else: 
       diag2 = False
    
    if vertical == True:
      moy_v = sum_v / nb_items

    if diag1 == True:
      moy_d1 = sum_d1 / nb_items

    if horizontal == True:
      moy_h = sum_h / nb_items

    if diag2 == True:
      moy_d2 = sum_d2 / nb_items
     
    return moy_v, moy_d1, moy_h, moy_d2
    

def convolve(image, Gm, Gd, stride=1, thres=1.0):
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
        
    # Get filter dimensions (Square)
    filter_dim = 5
    scan_dim = 2
    
    # Get image dimensions (Square)
    #nb_chan_img, img_dim, _ = image.shape 
    img_dim, _ = image.shape 
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
            #print("curr_x: %d. curr_y: %d. featMap_y: %d. featMap_x: %d." % (curr_x, curr_y, featMap_y, featMap_x ))
            moy_v, moy_d1, moy_h, moy_d2 = detecte_lines(Gm,Gd,scan_dim,thres,curr_y,curr_x)
            feature_maps [0, featMap_y, featMap_x] = moy_v
            feature_maps [1, featMap_y, featMap_x] = moy_d1
            feature_maps [2, featMap_y, featMap_x] = moy_h
            feature_maps [3, featMap_y, featMap_x] = moy_d2
            curr_x += stride
            featMap_x += 1
        curr_y += stride
        featMap_y += 1
    # Return all feature maps.
    return feature_maps
            
         


        Gm (Numpy ndarray): Gradient-intensed image to be processed
        Gd (Numpy ndarray): Gradient directions for each pixel in img  in the range [0, +180]
    Returns:
        Gm_nms (Numpy ndarray): Non-maximum suppression from Gradient magnitude
    """
    Gm_nms = np.copy(Gm)
    h,w = Gm.shape
    #x-coordinates : horizontal edges
    for x in range(1, w-1):
        #y-coordinates : vertical edges
        for y in range(1, h-1):
            #angle = 0
            if (Gd[y,x]<=22.5 or Gd[y,x]>157.5): dy, dx = 0, -1
            
            #angle = 45
            if (Gd[y,x]>22.5 and Gd[y,x]<=67.5): dy, dx = -1, 1
        
            #angle = 90
            if (Gd[y,x]>67.5 and Gd[y,x]<=112.5): dy, dx = -1, 0
                
            #angle = 135
            if (Gd[y,x]>112.5 and Gd[y,x]<=157.5): dy, dx = -1, -1
            
            for i in range(1, scan_dim):
                if Gm[y,x] <= Gm[y+dy*i,x+dx*i] and Gm[y,x] <= Gm[y-dy*i,x-dx*i]: Gm_nms[y,x]=0

    return Gm_nms

#TO DO : features map
def detecte_lines(Gm, Gd, scan_dim, thres, y, x):
    """Step 4: Detecte lines over the image following `Gradient Direction` and `Gradient Magnitude`.
    Args:
        param1 (int): The Gradient Magnitude of the image.
        param2 (int): The Gradient Direction of the image [0,+180].
        param3 (int): The dimension of the scan towards the up|down|right|left directions.
        param4 (int): The minimum value of the of the Gradient Magnitude.
        param5 (int): The x position from where to scan and detect lines.
        param6 (int): The y position from where to scan and detect lines.
    Returns:
        int: The Features map.
    """
    h,w = Gm.shape   
    
    #y-coordinates (North to South): vertical edges
    #x-coordinates (West to East): horizontal edges
    mag = Gm[y,x]
    
    vertical = True
    horizontal = True
    diag1 = True
    diag2 = True
    
    #angle = 0 
    if (Gd[y,x]<=22.5 or Gd[y,x]>157.5): 
      dy, dx = -1, 0 # +90 & -90
      if((y + scan_dim < h) and (y - scan_dim >= 0)):
        for i in list(range(1, scan_dim)) + list(range(-1, -scan_dim)):
          if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]>22.5 and Gd[y+dy*i,x+dx*i]<=157.5)):
            vertical = False
            break
      else: 
       vertical = False
       
    #angle = 45 
    if (Gd[y,x]>22.5 and Gd[y,x]<=67.5):  
      dy, dx = -1, -1 # +135 & -45
      if((x + scan_dim < w) and (y - scan_dim >= 0) and (x - scan_dim >= 0) and (y + scan_dim < h)):
       for i in list(range(1, scan_dim)) + list(range(-1, -scan_dim)):
        if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]<=22.5 and Gd[y+dy*i,x+dx*i]>67.5)):
         diag1 = False
         break
      else:
       diag1 = False  
       
    #angle = 90
    if (Gd[y,x]>67.5 and Gd[y,x]<=112.5):  
      dy, dx = 0, -1 # +180 & 0 
      if(x + scan_dim < w) and (x - scan_dim >= 0):
       for i in list(range(1, scan_dim)) + list(range(-1, -scan_dim)):
        if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]>=67.5 and Gd[y+dy*i,x+dx*i]>112.5)):
         horizontal = False
         break
      else: 
       horizontal = False
     
    #angle = 135
    if (Gd[y,x]>112.5 and Gd[y,x]<=157.5):  
      dy, dx = -1, 1 # +45 & -135
      if((x + scan_dim < w) and (y - scan_dim >= 0) and (x - scan_dim >= 0) and (y + scan_dim < h)):
       for i in list(range(1, scan_dim)) + list(range(-1, -scan_dim)):
        if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]<=112.5 and Gd[y+dy*i,x+dx*i]>157.5)):
         diag2 = False
         break
      else: 
       diag2 = False
     
    return (vertical or horizontal or diag1 or diag2)
    

def convolve(image, Gm, Gd, stride=1, thres=1.0):
    """Step 5: Confolve `Gradient filter` over `image` using `stride`

    Args:
        param1 (int): The image to convolve.
        param2 (int): The Gradient Magnitude of the image.
        param3 (int): The Gradient Direction of the image [0,+180]
        param4 (int): The stride that is parameter of the convolution.
        param5 (int): The minimum value of the of the Gradient Magnitude.

    Returns:
        int: The Features map.
    """
    
    nb_filter = 10
        
    # Get filter dimensions (Square)
    filter_dim = 5
    scan_dim = 2
    
    # Get image dimensions (Square)
    #nb_chan_img, img_dim, _ = image.shape 
    img_dim, _ = image.shape 
    print("img_dim: %d." % (img_dim))
	
    # Calculate output dimensions
    feature_dim = int((img_dim - filter_dim)/stride)+1 
    print("feature_dim: %d." % (feature_dim))
	
    # Initialize an empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros((nb_filter,feature_dim,feature_dim))
    
    # Convolve the image by the filter(s) over every part of the image, adding the bias at each step. 
    for curr_filter in range(nb_filter):
        curr_y = scan_dim
        featMap_y = 0
        while curr_y + scan_dim < img_dim:
            curr_x = scan_dim
            featMap_x = 0
            while curr_x + scan_dim < img_dim:
                #print("curr_x: %d. curr_y: %d. featMap_y: %d. featMap_x: %d." % (curr_x, curr_y, featMap_y, featMap_x ))
                feature_maps [curr_filter, featMap_y, featMap_x] = detecte_lines(Gm,Gd,scan_dim,thres,curr_y,curr_x)
                curr_x += stride
                featMap_x += 1
            curr_y += stride
            featMap_y += 1
    # Return all feature maps.
    return feature_maps
  
