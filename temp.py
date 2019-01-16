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