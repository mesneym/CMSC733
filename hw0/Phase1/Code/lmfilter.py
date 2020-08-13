import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.transform
from dogfilter import gauss2D



def lOG(sz,sigma):
    l = (sz-1)/2
    x,y = np.ogrid[-l:l+1:1,-l:l+1:1]
    var = sigma*sigma
    gauss2d = (1/(2*np.pi*sigma*sigma))*np.exp(-(x*x+y*y)/(2*var))
    log = (x*x + y*y - 2*var)/(var**2)*gauss2d
    return log

def gauss1D(sigma,sz,deg):
    l = (sz-1)/2
    x = np.arange(-l,l+1,1)
    var = sigma*sigma
    f = (1/((2*np.pi*var)**0.5)*np.exp(-(x*x)/(2*var)))

    if deg==0:
        return f
    elif deg==1:
        return -f*(x/var)
    else:
        return f*((x*x-var)/var**2)

def skewedGauss2D(sigma,l,degX,degY):
    row = gauss1D(sigma,l,degY)
    col = gauss1D(3*sigma,l,degX)
    return np.outer(row,col)
     

def lmFilterBank(scale,sz):
    # returns a filter bank of size [szxszx48]
    F = np.zeros([sz,sz,48])
    
    #Edge and bar filters
    count = 0
    orient = np.linspace(0,180,7)
    for i in range(3):
        for j in range(6):
            kernel = skewedGauss2D(scale[i],sz,0,1)
            F[:,:,j+12*i] = skimage.transform.rotate(kernel,orient[j])
            kernel = skewedGauss2D(scale[i],sz,0,2)
            F[:,:,j+6+12*i] = skimage.transform.rotate(kernel,orient[j])
            count += 1

    #log filter
    for i in range(4):
        F[:,:,36+i] = lOG(sz,scale[i])
        F[:,:,36+i+4] = lOG(sz,3*scale[i])

    #gaussian filter
    for i in range(4):
        F[:,:,44+i] =gauss2D(sz,scale[i]) 
    return F



if __name__=='__main__':
    scale = [1,2**0.5,2,2*(2**0.5)]
    F = lmFilterBank(scale=scale,sz=49)

    for i in range(48):
        plt.subplot(4,12,i+1)
        plt.axis('off')
        plt.imshow(F[:,:,i],cmap='gray')
    plt.show()


