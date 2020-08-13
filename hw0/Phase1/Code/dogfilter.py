import numpy as np
import cv2
import skimage.transform
import matplotlib.pyplot as plt


def gauss2D(sz,sigma):
    l = (sz-1)/2
    x,y = np.ogrid[-l:l+1:1,-l:l+1:1]
    var = sigma*sigma
    f = (1/(2*np.pi*sigma*sigma))*np.exp(-(x*x+y*y)/(2*var))
    return f

def dogFilterBank(norient,scale,sz):
    #returns filter of size sz x sz x (norient x scale)

    F = np.zeros([sz,sz,norient*len(scale)])
    count = 0
    for s in scale:
        constant = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
        kernel = gauss2D(sz,s)
        dG = cv2.Sobel(kernel, cv2.CV_64F, 1, 0, ksize=3,borderType=constant)
        orient = np.linspace(0,360,norient+1)
        for j in range(len(orient)-1):
            f = skimage.transform.rotate(dG,orient[j])
            F[:,:,count] = f
            count += 1
    return F        


if __name__ == '__main__':
    # scale = [9,16,25]
    # orient= 15
    # size = 49
    scale = [4,8] 
    orient= 16
    size = 49
    F = dogFilterBank(orient,scale,49)
    
    l = len(scale)*orient
    for i in range(0,l,1):
        plt.subplot(len(scale),orient,i+1)
        plt.axis('off')
        plt.imshow(F[:,:,i],cmap='gray')
    
    plt.show()




