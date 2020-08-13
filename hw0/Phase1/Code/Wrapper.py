#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code
"""

# Code starts here:

import numpy as np
import cv2
import skimage.transform
import sklearn.cluster
import matplotlib.pyplot as plt
from dogfilter import dogFilterBank  
from lmfilter import lmFilterBank    
from gaborfilter import gaborFilterBank    
from halfdiskmask import hDiskMask

def filterConvImg(img,nf,F):
    results = np.array(img)
    for i in range(nf):
        if(type(F) == list):
            dst = cv2.filter2D(img,cv2.CV_32F,F[i])
        else:
            dst = cv2.filter2D(img,cv2.CV_32F,F[:,:,i])
        results = np.dstack((results,dst))
    return results


def clusterID(data,numClusters):
    if(len(data.shape) == 3):
        i,j,k = data.shape
    else:
        i,j = data.shape
        k = 1

    reshapedData = data.reshape(((i*j),k)) #row->image pixels columns-> k features
    kmeans = sklearn.cluster.KMeans(n_clusters = numClusters, random_state = 2)
    kmeans.fit(reshapedData)
    labels = kmeans.labels_
    labels = labels.reshape((i,j)) #Id associated with pixel
    return labels


def chiSqr(img,lmask,rmask,bins):
    chiSqrdist = (img*0).astype(float)
    for i in range(bins):
        tmp = (img == i).astype(int)
        g = cv2.filter2D(tmp,cv2.CV_64F,lmask)
        h = cv2.filter2D(tmp,cv2.CV_64F,rmask)
        numerator = (g-h)**2 
        denom = g+h
        chiSqrdist += np.divide(numerator,denom,out=np.zeros_like(numerator),where=denom!=0) 
    return 0.5*chiSqrdist


def gradient(img,hmask,bins):
    grad = img
    n = int(len(hmask)/2)
    for i in range(n):
        g = chiSqr(img,hmask[2*i],hmask[2*i+1],bins)
        grad = np.dstack((grad,g))
    return np.mean(grad,axis=2)



def plot(images,titles,fig,color):
    for i in range(4):
        ax = fig.add_subplot(2,2,i+1)
        ax.set_title(titles[i])
        plt.imshow(images[i],cmap=color[i])
        plt.axis('off')

def main():
        #############################
        #        parameters 
        #############################
        #Hdisk parameters
        scaleHmask = [4,8,12]; norientHmask = 8
        
        #number of clusters for texture,brightness and scale
        nClustersT = 64; nClustersB = 16 ; nClustersC = 16

        #weights for pblite
        w1 = w2 = 0.5

        #dog filter params
        norientDog = 16; scaleDog = [4,8] ; szDog = 49

        #lm filter params
        scaleLm =[1,2**0.5,2,2*(2**0.5)] ; szLm = 49

        #gabor filter params
        sigmaG=[9,16,25]; thetaG=0.25; LambdaG=1; psiG=1; gammaG=1; nfG=15


        for i in range(10):
            imgPath = '/home/ak/CMSC-733/hw0/Phase1/BSDS500/Images/'+str(i+1) +'.jpg'
            img = cv2.imread(imgPath)
            imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            """
            Generate Difference of Gaussian Filter Bank: (DoG)
            Display all the filters in this filter bank and save image as DoG.png,
            use command "cv2.imwrite(...)"
            """
            dogF = dogFilterBank(norient=norientDog,scale=scaleDog,sz=szDog)


            """
            Generate Leung-Malik Filter Bank: (LM)
            Display all the filters in this filter bank and save image as LM.png,
            use command "cv2.imwrite(...)"
            """
            lmF = lmFilterBank(scale = scaleLm,sz=szLm)


            """
            Generate Gabor Filter Bank: (Gabor)
            Display all the filters in this filter bank and save image as Gabor.png,
            use command "cv2.imwrite(...)"
            """
            gbF =  gaborFilterBank(sigma=sigmaG, theta=thetaG, Lambda=LambdaG, psi=psiG, gamma=gammaG,nf=nfG)


            """
            Generate Texton Map
            Filter image using oriented gaussian filter bank
            """
            dogFConvImg = filterConvImg(imgGray,dogF.shape[2],dogF)
            lmFConvImg = filterConvImg(imgGray,lmF.shape[2],lmF)
            gbFConvImg = filterConvImg(imgGray,len(gbF),gbF)
            textonMap = np.dstack((dogFConvImg[:,:,1],lmFConvImg[:,:,1],gbFConvImg[:,:,1])) 

            """
            Generate texture ID's using K-means clustering
            Display texton map and save image as TextonMap_ImageName.png,
            use command "cv2.imwrite('...)"
            """
            T=clusterID(textonMap,nClustersT)

            """
            Generate Brightness Map
            Perform brightness binning 
            """
            B=clusterID(imgGray,nClustersB)

            """
            Generate Color Map
            Perform color binning or clustering
            """
            C= clusterID(img,nClustersC)

            """  
            Display plots for texture brightness and color
            """ 
            fig1 = plt.figure()
            fig1.suptitle("Image Texture")
            titles = ["Original image","Brightness","Texture","Color"]
            images = [img,B,T,C]
            color = ['viridis','viridis','viridis','viridis']
            plot(images,titles,fig1,color) 
            fig1.savefig('./../results/'+str(i+1)+'_texture.png')
            plt.close(fig1)

            """
            Generate Half-disk masks
            Display all the Half-disk masks and save image as HDMasks.png,
            use command "cv2.imwrite(...)"
            """
            hdisk = hDiskMask(scaleHmask,norientHmask)

       
            """
            Generate Texton Gradient (Tg)
            Perform Chi-square calculation on Texton Map
            Display Tg and save image as Tg_ImageName.png,
            use command "cv2.imwrite(...)"
            *** 
            """
            Tg = gradient(T,hdisk,nClustersT) 

        
            """
            Generate Brightness Gradient (Bg)
            Perform Chi-square calculation on Brightness Map
            Display Bg and save image as Bg_ImageName.png,
            use command "cv2.imwrite(...)"
            """
            Bg = gradient(B,hdisk,nClustersB) 
        
        
            """
            Generate Color Gradient (Cg)
            Perform Chi-square calculation on Color Map
            Display Cg and save image as Cg_ImageName.png,
            use command "cv2.imwrite(...)"
            """
            Cg = gradient(C,hdisk,nClustersC) 
        
            """
            Displaying texture,color and brightness gradients
            """
            fig2 = plt.figure()
            fig2.suptitle("Image Texture Gradients")
            titles = ["Original image","Brightness Gradient","Texture Gradient","Color Gradient"]
            images = [img,Bg,Tg,Cg]
            color = ['viridis','viridis','viridis','viridis']
            plot(images,titles,fig2,color) 
            fig2.savefig('./../results/'+str(i+1)+'_gradients.png')
            plt.close(fig2)
        

            """
            Read Sobel Baseline
            use command "cv2.imread(...)"
            """
            sobelPb = cv2.imread('../BSDS500/SobelBaseline/'+str(i+1)+'.png',0)

            """
            Read Canny Baseline
            use command "cv2.imread(...)"
            """
            cannyPb = cv2.imread('../BSDS500/CannyBaseline/'+str(i+1)+'.png',0)
            
            
            """
            Combine responses to get pb-lite output
            Display PbLite and save image as PbLite_ImageName.png
            use command "cv2.imwrite(...)"
            """
            pbEdges = ((Tg+Bg+Cg)/3)*(w1*cannyPb+w2*sobelPb)

            """
            Displaying plots
            """
            fig3 = plt.figure()
            fig3.suptitle("Edges - PBLite vs Baselines")
            titles = ["Original image","pbEdges","sobel Baseline","Canny Baseline"]
            images = [img,pbEdges,sobelPb,cannyPb]
            color = ['viridis','gray','gray','gray']
            plot(images,titles,fig3,color) 
            fig3.savefig('./../results/'+str(i+1)+'_edges.png')
            plt.close(fig3)
            
            if cv2.waitKey(0) & 0XFF == ord('q'):
                break
    
if __name__ == '__main__':
    main()



