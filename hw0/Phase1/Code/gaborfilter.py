
#Code obtained from wiki

import numpy as np
import skimage.transform
import matplotlib.pyplot as plt


def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def gaborFilterBank(sigma,theta,Lambda,psi,gamma,nf):
    angle = np.linspace(0,360,nf)
    F = []

    for s in sigma:
        g = gabor(s,theta,Lambda,psi,gamma)
        for i in range(nf):
            f = skimage.transform.rotate(g,angle[i])
            F.append(f)
    return F

if __name__=='__main__':
    F = gaborFilterBank(sigma=[9,16,25], theta=0.25, Lambda=1, psi=1, gamma=1,nf=15)
    l = len(F) 

    for i in range(l):
        plt.subplot(l/5,5,i+1)
        plt.axis('off')
        plt.imshow(F[i],cmap='gray')

    plt.show() 


