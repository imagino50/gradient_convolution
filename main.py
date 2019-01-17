import matplotlib.pyplot as plt 
from core import *

# Image Input 
#image = data.camera() 
#image = data.coins() 
image = data.horse() 
print ("image.shape: ", ', '.join(map(str, image.shape)))

Gm, Gd = compute_gradient(image)
Gm_nms = supress_non_max(Gm, Gd, scan_dim=2, thres=0.4)
feature_maps = convolve(image, Gm_nms, Gd, stride=1, thres=0.8) 
            
plt.figure("SOBEL EDGES DETECTION") 
plt.subplot(1,5,1) 
plt.imshow(image, interpolation='nearest') 
plt.title("IMAGE") 
plt.subplot(1,5,2) 
plt.imshow(Gm, interpolation='nearest') 
plt.title("MAGNITUDES") 
plt.subplot(1,5,3) 
plt.imshow(Gd, interpolation='nearest') 
plt.title("ANGLES") 
plt.subplot(1,5,4) 
plt.imshow(Gm_nms, interpolation='nearest') 
plt.title("ANGLES NMS") 
plt.subplot(1,5,5) 
plt.imshow(feature_maps[0], interpolation='nearest') 
plt.title("feature_maps[0]") 
plt.show()
