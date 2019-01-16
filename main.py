
            
            
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
