import numpy as np
import matplotlib.pyplot as plt
import skimage.transform


def structuralElements(r):
    #returns circle and rectangular structural elements 
    x,y = np.mgrid[-r:r+1:1,-r:r+1:1]
    circle = x*x+y*y<=r*r
    rhalfplane = y>0 
    lhalfplane = y<0
    return circle,rhalfplane,lhalfplane

def hDiskMask(scale,norient):
 
    hdisk = []
    orient = np.linspace(0,180,norient+1)
    
    for s in scale:
        circle,rhalfPlane,lhalfPlane = structuralElements(s)

        for i in range(len(orient)-1): 
            rRotHalfPlane = skimage.transform.rotate(rhalfPlane,orient[i])
            lRotHalfPlane = skimage.transform.rotate(lhalfPlane,orient[i])

            firstHalfDisk = np.logical_and(rRotHalfPlane,circle).astype(int)
            secondHalfDisk = np.logical_and(lRotHalfPlane,circle).astype(int)
             
            hdisk.append(secondHalfDisk)
            hdisk.append(firstHalfDisk)

    return hdisk

if __name__=='__main__':
    scale = [4,8,12]
    norient = 8
    hdisk = hDiskMask(scale,norient)
   
    n = len(scale)* norient*2

    for i in range(n):
        plt.subplot(len(scale),norient*2,i+1)
        plt.axis('off')
        plt.imshow(hdisk[i],cmap='gray')
    
    plt.show()



