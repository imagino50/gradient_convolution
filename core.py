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

### TODO : Padding + maxnonsuppres 5*5 + cross edge detection

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


def detecte_lines(Gm, Gd, scan_dim=2, thres=1.0, y, x):
    """Step 4: Detecte lines over the image using `Gradient Direction` and `Gradient Magnitude`.
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
	
	#y-coordinates : vertical edges
	#x-coordinates : horizontal edges
	mag = Gm[y,x]  
	
	#angle = 0 
	if (Gd[y,x]<=22.5 or Gd[y,x]>157.5): 
	  dy, dx = -1, 0 # +90 & -90
	  if(y + scan_dim < h)
	    for i in range(1, scan_dim):
	      if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]>22.5 and Gd[y+dy*i,x+dx*i]=<157.5))
	  	    nn = false
	      else 
	        nn = false
      
	  if(y - scan_dim => 0)
	   for i in range(1, scan_dim):
	    if(Gm[y-dy*i,x-dx*i]<thres or (Gd[y-dy*i,x-dx*i]>22.5 and Gd[y-dy*i,x-dx*i]=<157.5))
	  	ss = false   
	  else 
	   ss = false
	
	#angle = 45
	if (Gd[y,x]>22.5 and Gd[y,x]<=67.5):  
	  dy, dx = -1, -1 # +135 & -45
	  if(x + scan_dim < w) and (y - scan_dim => 0)  
	   for i in range(1, scan_dim):
	    if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]=<22.5 and Gd[y+dy*i,x+dx*i]>67.5))
	  	nw = false
	  else 
	   nw = false
      
	  if(x - scan_dim => 0) and (y + scan_dim < h) 
	   for i in range(1, scan_dim):
	    if(Gm[y-dy*i,x-dx*i]<thres or (Gd[y-dy*i,x-dx*i]=<22.5 and Gd[y-dy*i,x-dx*i]>67.5))
	  	se = false   
	  else 
	   se = false
	
	#angle = 90
	if (Gd[y,x]>67.5 and Gd[y,x]<=112.5):  
	  dy, dx = 0, -1 # +180 & 0 
	  if(x + scan_dim < w) 
	   for i in range(1, scan_dim):
	    if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]=>67.5 and Gd[y+dy*i,x+dx*i]>112.5))
	  	ww = false
	  else 
	   ww = false
      
	  if(x - scan_dim => 0) 
	   for i in range(1, scan_dim):
	    if(Gm[y-dy*i,x-dx*i]<thres or (Gd[y-dy*i,x-dx*i]=>67.5 and Gd[y-dy*i,x-dx*i]>112.5))
	  	ee = false   
	  else 
	   ee = false
	
	#angle = 135
	if (Gd[y,x]>112.5 and Gd[y,x]<=157.5):  
	  dy, dx = -1, 1 # +45 & -135
	  if(x + scan_dim < w) and (y - scan_dim => 0)  
	   for i in range(1, scan_dim):
	    if(Gm[y+dy*i,x+dx*i]<thres or (Gd[y+dy*i,x+dx*i]=<112.5 and Gd[y+dy*i,x+dx*i]>157.5))
	  	ne = false
	  else 
	   ne = false
      
	  if(x - scan_dim => 0) and (y + scan_dim < h) 
	   for i in range(1, scan_dim):
	    if(Gm[y-dy*i,x-dx*i]<thres or (Gd[y-dy*i,x-dx*i]=<112.5 and Gd[y-dy*i,x-dx*i]>157.5))
	  	sw = false   
	  else 
	   sw = false
	 
	return nms   
 
	

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
    nb_chan_img, img_dim, _ = image.shape 
    
    # Calculate output dimensions
    feature_dim = int((img_dim - filter_dim)/stride)+1 
        
    # Initialize an empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros(nb_filter,feature_dim,feature_dim)
    
    # Convolve the image by the filter(s) over every part of the image, adding the bias at each step. 
    for curr_filter in range(nb_filter):
	    curr_y = scan_dim
        featMap_y = 0
        while curr_y + scan_dim <= img_dim:
		    curr_x = scan_dim
            featMap_x = 0
            while curr_x + scan_dim <= img_dim:
                # Get the current region to get multiplied with the current filter.
                #curr_region = image[:,curr_y:curr_y+filter_dim, curr_x:curr_x+filter_dim]
                # Sum the result of multiplication between the current region and the current filter.
                #feature_maps [curr_filter, featMap_y, featMap_x] = np.sum(filter[curr_filter] * curr_region) + bias[curr_filter]
				feature_maps [curr_filter, featMap_y, featMap_x] = linesDetection(Gm,Gd,scan_dim,thres,curr_y,curr_x)
                curr_x += stride
                featMap_x += 1
            curr_y += stride
            featMap_y += 1
    # Return all feature maps.
    return feature_maps
            
         

