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
import matplotlib.pyplot as plt 
import numpy as np

# Image Input 
#image = data.camera() 
#image = data.coins() 
image = data.horse() 
print ("image.shape: ", ', '.join(map(str, image.shape)))

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
	
#edge_magnitues = (sob_x ** 2 + sobel_y ** 2) ** .5 
#edges = filters.sobel(camera) 
#gradient_dir = ((np.arctan(gy/gx))/np.pi) * 180 # radian to degree conversion 
#Convert angles from radians to degrees.
#rad2deg(Angle in radians)
#Return the corresponding angle in degrees.
#angle = np.rad2deg(angle) % 180

"""Keep the edge angle of an image with a Magnitude minimum. """ 
#MIN_MAG = 0.08 
#edge_angles2 = np.copy(gradient_direction) 
#edge_angles2[edge_magnitues < MIN_MAG] = 0 
#edge_angles = np.multiply(dir, 180/math.pi) 
	
"""Convolve using the edge angle. """ 
#print ("conv1.shape: ", ', '.join(map(str, conv1.shape)))


#https://github.com/fubel/PyCannyEdge/blob/master/CannyEdge/core.py
def supress_non_max(Gm, Gd, th=1.0):
    """ Step 3: Non-maximum suppression
    Args:
        Gm (Numpy ndarray): Gradient-intensed image to be processed
        Gd (Numpy ndarray): Gradient directions for each pixel in img  in the range [0, +180]
    Returns:
        nms: Numpy ndarray of Non-maximum suppression
    """
	nms = np.copy(Gm)
	h,w = Gm.shape
	#x-coordinates : horizontal edges
	for x in range(1, w-1):
		#y-coordinates : vertical edges
		for y in range(1, h-1):      
			if (Gd[y,x]<=22.5 or Gd[y,x]>157.5): #angle = 0
				if(Gm[y,x]<=Gm[y,x-1]) and (Gm[y,x]<=Gm[y,x+1]): nms[y,x]=0
			if (Gd[y,x]>22.5 and Gd[y,x]<=67.5): #angle = 45
				if(Gm[y,x]<=Gm[y-1,x+1]) and (Gm[y,x]<=Gm[y+1,x-1]): nms[y,x]=0
			if (Gd[y,x]>67.5 and Gd[y,x]<=112.5): #angle = 90
				if(Gm[y,x]<=Gm[y-1,x]) and (Gm[y,x]<=Gm[y+1,x]): nms[y,x]=0
			if (Gd[y,x]>112.5 and Gd[y,x]<=157.5): #angle = 135
				if(Gm[y,x]<=Gm[y-1,x+1]) and (Gm[y,x]<=Gm[y+1,x-1]): nms[y,x]=0
	return nms
			
			

## nonmaximum suppression
# Gm: gradient magnitudes
# Gd: gradient directions, -pi/2 to +pi/2
# return: nms, gradient magnitude if local max, 0 otherwise
def nonmaxsupress(Gm, Gd, th=1.0):
	#Gd: [-pi/2, +pi/2]
	#Gd[Gd > 0.5*numpy.pi] -= numpy.pi
    #Gd[Gd < -0.5*numpy.pi] += numpy.pi
	
    nms = zeros(Gm.shape, Gm.dtype)   
    h,w = Gm.shape    
    for x in range(1, w-1):
        for y in range(1, h-1):            
            mag = Gm[y,x]
            if mag < th: continue        
            teta = Gd[y,x]            
            dx, dy = 0, -1      # abs(orient) >= 1.1781, teta < -67.5 degrees and teta > 67.5 degrees
            if abs(teta) <= 0.3927: dx, dy = 1, 0       # -22.5 <= teta <= 22.5
            elif teta < 1.1781 and teta > 0.3927: dx, dy = 1, 1     # 22.5 < teta < 67.5 degrees
            elif teta > -1.1781 and teta < -0.3927: dx, dy = 1, -1  # -67.5 < teta < -22.5 degrees            
            if mag > Gm[y+dy,x+dx] and mag > Gm[y-dy,x-dx]: nms[y,x] = mag    
			#if mag > Gm[y+dy,x+dx] and mag > Gm[y-dy,x-dx] and mag > Gm[y+dy*2,x+dx*2] and mag > Gm[y-dy*2,x-dx*2]: nms[y,x] = mag
			#for i in range(1, scan_dim):
			 #if mag =< Gm[y+dy*i,x+dx*1] or mag =< Gm[y-dy*i,x-dx*i]:
			  
			  
    return nms


# TODO : Padding + maxnonsuppres 5*5 + cross edge detection
def detecte_lines(Gm, Gd, scan_dim=2, th=1.0, y, x):
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
	mag = Gm[y,x]  
	
	#angle = 0 
	if (Gd[y,x]<=22.5 or Gd[y,x]>157.5): 
	  dx, dy = 0, -1 # +90 & -90
	  if(y + scan_dim < h)
	    for i in range(1, scan_dim):
	      if(Gm[y+dy*i,x+dx*i]<th or (Gd[y+dy*i,x+dx*i]>22.5 and Gd[y+dy*i,x+dx*i]=<157.5))
	  	    nn = false
	      else 
	        nn = false
      
	  if(y - scan_dim => 0)
	   for i in range(1, scan_dim):
	    if(Gm[y-dy*i,x-dx*i]<th or (Gd[y-dy*i,x-dx*i]>22.5 and Gd[y-dy*i,x-dx*i]=<157.5))
	  	ss = false   
	  else 
	   ss = false
	
	#angle = 45
	if (Gd[y,x]>22.5 and Gd[y,x]<=67.5):  
	  dx, dy = -1, -1 # +135 & -45
	  if(x + scan_dim < w) and (y - scan_dim => 0)  
	   for i in range(1, scan_dim):
	    if(Gm[y+dy*i,x+dx*i]<th or (Gd[y+dy*i,x+dx*i]=<22.5 and Gd[y+dy*i,x+dx*i]>67.5))
	  	nw = false
	  else 
	   nw = false
      
	  if(x - scan_dim => 0) and (y + scan_dim < h) 
	   for i in range(1, scan_dim):
	    if(Gm[y-dy*i,x-dx*i]<th or (Gd[y-dy*i,x-dx*i]=<22.5 and Gd[y-dy*i,x-dx*i]>67.5))
	  	se = false   
	  else 
	   se = false
	
	#angle = 90
	if (Gd[y,x]>67.5 and Gd[y,x]<=112.5):  
	  dx, dy = -1, 0 # +180 & 0
	  if(x + scan_dim < w) 
	   for i in range(1, scan_dim):
	    if(Gm[y+dy*i,x+dx*i]<th or (Gd[y+dy*i,x+dx*i]=>67.5 and Gd[y+dy*i,x+dx*i]>112.5))
	  	ww = false
	  else 
	   ww = false
      
	  if(x - scan_dim => 0) 
	   for i in range(1, scan_dim):
	    if(Gm[y-dy*i,x-dx*i]<th or (Gd[y-dy*i,x-dx*i]=>67.5 and Gd[y-dy*i,x-dx*i]>112.5))
	  	ee = false   
	  else 
	   ee = false
	
	#angle = 135
	if (Gd[y,x]>112.5 and Gd[y,x]<=157.5):  
	  dx, dy = 1, -1 # +45 & -135
	  if(x + scan_dim < w) and (y - scan_dim => 0)  
	   for i in range(1, scan_dim):
	    if(Gm[y+dy*i,x+dx*i]<th or (Gd[y+dy*i,x+dx*i]=<112.5 and Gd[y+dy*i,x+dx*i]>157.5))
	  	ne = false
	  else 
	   ne = false
      
	  if(x - scan_dim => 0) and (y + scan_dim < h) 
	   for i in range(1, scan_dim):
	    if(Gm[y-dy*i,x-dx*i]<th or (Gd[y-dy*i,x-dx*i]=<112.5 and Gd[y-dy*i,x-dx*i]>157.5))
	  	sw = false   
	  else 
	   sw = false
	 
	return nms   
 
	

def convolve(image, Gm, Gd, stride=1, th=1.0):
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
				feature_maps [curr_filter, featMap_y, featMap_x] = linesDetection(Gm,Gd,scan_dim,th,curr_y,curr_x)
                curr_x += stride
                featMap_x += 1
            curr_y += stride
            featMap_y += 1
    # Return all feature maps.
    return feature_maps
            
         
            
            
plt.figure("SOBEL EDGES DETECTION") 
plt.subplot(1,4,1) 
plt.imshow(image, interpolation='nearest') 
plt.title("IMAGE") 
plt.subplot(1,4,2) 
plt.imshow(edge_magnitues, interpolation='nearest') 
plt.title("MAGNITUDES") 
plt.subplot(1,4,3) 
plt.imshow(edge_angles, interpolation='nearest') 
plt.title("ANGLES") 
plt.subplot(1,4,4) 
plt.imshow(edge_angles2, interpolation='nearest') 
plt.title("ANGLES THRES") 
plt.show()
