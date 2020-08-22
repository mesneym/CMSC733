#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

"""



#@TODO Tweaking and accuracy

# Add any python libraries here
import numpy as np
import cv2
import copy



def readImages():
    path = './../Data/Train/Set'
    imgSets = {}
    n = 3
    for i in range(3):
        setPath = path+str(i+1)
        imgSets[i] = []
        if(i == 2):  n = 8
        for j in range(n):
            imgPath = setPath + '/' +str(j+1)+'.jpg'
            img = cv2.imread(imgPath)
            imgSets[i].append(img)
    return imgSets


def displayCorners(img,features,path):
    imgCopy = copy.deepcopy(img)
    for f in features:
        x,y = f.ravel(order='F')
        cv2.circle(imgCopy,(x,y),2,(0,0,255),-1)
    cv2.imwrite(path,imgCopy)
    return imgCopy

def getAndDisplayCorners(img,path):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(imgGray, 150000, 0.01,15)
    imgCopy = displayCorners(img,features,path) 
    return imgCopy,features

def ANMS(img,features,nbest,path):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bestFeatures = []
    nfeatures,_,_ = features.shape
    r = np.ones(nfeatures)*np.inf
    ED = np.inf
   
    for i in range(nfeatures):
        xi,yi = features[i].ravel().astype(np.int)
        for j in range(nfeatures):
            xj,yj = features[j].ravel().astype(np.int)
            if(imgGray[yj,xj]>imgGray[yi,xi]):
                ED = (xi-xj)**2 + (yi-yj)**2 #measures how close features are to strong pts
            if(ED<r[i]):
                r[i]=ED
    
    bestFeatures=features[r.argsort()[::-1],:,:] #select features that are further away from strong pts
    imgCopy = displayCorners(img,bestFeatures[0:nbest,:,:],path)
    return imgCopy,bestFeatures[0:nbest,:,:]
    

def descriptor(img,features,sz,path):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fdesc = []
    offset = int(sz/2)
    nfeatures = features.shape[0]
    padImg = np.pad(imgGray,offset,'constant',constant_values=0)
    
    for i in range(nfeatures):
        x,y = features[i].ravel()
        featuresX,featuresY = (np.array([x,y])+ [offset,offset]).astype(np.int)

        patch = padImg[featuresY-offset:featuresY+offset,featuresX-offset:featuresX+offset]
        blurPatch = cv2.GaussianBlur(patch,(5,5),cv2.BORDER_DEFAULT) 
        sample = blurPatch[0::5,0::5]

        vec = sample.reshape((int(sz/5)**2,1))
        vec = vec - np.mean(vec)
        vec = vec/np.std(vec)
        fdesc.append([np.array([x,y]),vec])

    # cv2.imwrite(path,fdesc[0])
    return fdesc

def match(fdesc1,fdesc2):
    pairs = []
    for i in range(len(fdesc1)):
        vec1 = fdesc1[i][1]
        dist = np.zeros(len(fdesc2))
        for j in range(len(fdesc2)):
            vec2 = fdesc2[j][1]
            dist[j] = np.linalg.norm(vec1-vec2)**2

        ind = dist.argsort()
        if(dist[ind[0]]/dist[ind[1]]<0.45):
            pairs.append([fdesc1[i][0],fdesc2[ind[0]][0]])
    return pairs


def drawPairs(img1,img2,matchPairs,path):
    img  = np.hstack((img1,img2))
    for i in range(len(matchPairs)):
        x1,y1 = (matchPairs[i][0]).astype(np.int)
        x2,y2 = (matchPairs[i][1]).astype(np.int)
        x2 = x2+img1.shape[1]

        cv2.line(img,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.circle(img,(x1,y1),3,(0,0,225),1)
        cv2.circle(img,(x2,y2),3,(0,0,225),1)
        
    cv2.imwrite(path,img)
    return img


def ransacHomography(P,threshold,iterations,acc):
    maxInliers = -1*np.inf
    bestinliers = []

    for i in range(iterations):
        r = np.random.randint(0,len(P),4)
        p1 = np.array([P[r[0]][0],P[r[1]][0],P[r[2]][0],P[r[3]][0]]) 
        p2 = np.array([P[r[0]][1],P[r[1]][1],P[r[2]][1],P[r[3]][1]]) 
        H_1_2= cv2.getPerspectiveTransform(p2,p1) 
        inliers = []

        for j in range(len(P)):
            p1 = P[j][0]; p2 = P[j][1]
            pp1 = H_1_2.dot([p2[0],p2[1],1])
            if(pp1[2]==0):continue
            pp1 = pp1/pp1[2]

            SSD = np.linalg.norm(p1 - pp1[0:2])
            if(SSD<threshold):
                inliers.append(j)

        if(len(inliers)>maxInliers):
            maxInliers = len(inliers)
            bestinliers = inliers

            if(len(inliers)>acc*len(P)): break
    
    if(len(bestinliers)<5):
        print(len(bestinliers))
        return None,None,False

    bestPairs = [P[i] for i in bestinliers]
    p1 = np.array([P[i][0] for i in bestinliers])
    p2 = np.array([P[i][1] for i in bestinliers])
    H_1_2,_ = cv2.findHomography(p2,p1,0)
    return H_1_2,bestPairs,True
    
def coordinateTransformation(x,xmin,xmax,desiredXmin,desiredXmax):
    Tx =  ((x-xmin)/(xmax-xmin))*(desiredXmax-desiredXmin) + desiredXmin 
    return Tx


#@TODO warpBlend- make robust
def warpBlend(H_1_2,img1,img2,path):
    y2,x2,_ = img2.shape
    y1,x1,_ = img1.shape
    p2 = np.array([[x2,x2,0,0],[y2,0,y2,0],[1,1,1,1]])
    p1 = np.array([[x1,x1,0,0],[y1,0,y1,0],[1,1,1,1]])
    pp1 = H_1_2.dot(p2)
    pp1 = (pp1/pp1[2,:]).astype(np.int)
    N = np.hstack((pp1,p1))

    wImg2Xmin, wImg2Xmax = min(pp1[0,:]),max(pp1[0,:])
    wImg2Ymin, wImg2Ymax = min(pp1[1,:]),max(pp1[1,:])
    xmin,xmax,ymin,ymax = min(N[0,:]),max(N[0,:]),min(N[1,:]),max(N[1,:])

    blendImg = np.zeros((ymax-ymin,xmax-xmin,3),np.uint8)

    ToriginX = int(coordinateTransformation(0,xmin,xmax,0,xmax-xmin))
    ToriginY = int(coordinateTransformation(0,ymin,ymax,0,ymax-ymin))
    Tx1 = ToriginX+x1 
    Ty1 = ToriginY+y1


    TwImg2Xmax = int(coordinateTransformation(wImg2Xmax,xmin,xmax,0,xmax-xmin))
    TwImg2Xmin = int(coordinateTransformation(wImg2Xmin,xmin,xmax,0,xmax-xmin))
    TwImg2Ymax = int(coordinateTransformation(wImg2Ymax,ymin,ymax,0,ymax-ymin))
    TwImg2Ymin = int(coordinateTransformation(wImg2Ymin,ymin,ymax,0,ymax-ymin)) 
    
    Translate = np.array([[1,0,-wImg2Xmin+TwImg2Xmin],[0,1,-wImg2Ymin+TwImg2Ymin],[0,0,1]])
    warpedImg=cv2.warpPerspective(img2,Translate.dot(H_1_2),(int(xmax-xmin),int(ymax-ymin)))
  
   
    distY = wImg2Ymax-wImg2Ymin; distX = wImg2Xmax-wImg2Xmin
    blendImg[ToriginY:Ty1,ToriginX:Tx1]= img1
    
    indices = np.where(np.any(warpedImg != [0,0,0],axis=-1))
    blendImg[indices[0],indices[1],:] = warpedImg[indices[0],indices[1],:]
    
    cv2.imwrite(path,blendImg)
    return blendImg


def main():
      
      """
      Read a set of images for Panorama stitching
      """ 
      imgSet = readImages()
      stitchedImg = []
      
      for s in range(3):
          path = './results/set' + str(s) +'/'

          for i in range(0,len(imgSet[s])-1):
              if(i==0):img1 = cv2.resize(imgSet[s][i],(500,500))
              else: img1 = cv2.resize(stitchedImg,(500,500))
              img2 = cv2.resize(imgSet[s][i+1],(500,500))

              """
              Corner Detection
              Save Corner detection output as corners.png
              """
              if(i==0): _,features1 = getAndDisplayCorners(img1,path+str(i)+'_corner.png')
              else: _,features1 = getAndDisplayCorners(img1,path+str(i)+'_stitched_corner.png')
              _,features2 = getAndDisplayCorners(img2,path+str(i+1)+'_corner.png')

              """
              Perform ANMS: Adaptive Non-Maximal Suppression
              Save ANMS output as anms.png
              """
              if(i==0): _,bestFeatures1 = ANMS(img1,features1,300,path+str(i)+'_ANMScorner.png')
              else: _,bestFeatures1 = ANMS(img1,features1,300,path+str(i)+'_stitched_ANMScorner.png')
              _,bestFeatures2 = ANMS(img2,features2,300,path+str(i+1)+'_ANMScorner.png')

              
              """
              Feature Descriptors
              Save Feature Descriptor output as FD.png
              """
              if(i==0): fdesc1 = descriptor(img1,bestFeatures1,40,path+str(i)+'FD.png')
              else: fdesc1 = descriptor(img1,bestFeatures1,40,path+str(i)+'_stitched_FD.png')
              fdesc2 = descriptor(img2,bestFeatures2,40,path+str(i+1)+'FD.png')
              # cv2.imshow('featureDescriptor0',fdesc1[0][1])

              
              """
              Feature Matching
              Save Feature Matching output as matching.png
              """
              pairs = match(fdesc1,fdesc2)
              imgMatch = drawPairs(img1,img2,pairs,path+str(i+1)+'_matching.png')
              # cv2.imshow('featureMatch',imgMatch)


              """
              Refine: RANSAC, Estimate Homography
              """
              H,ransacPairs,success = ransacHomography(pairs,18,2000,0.9)
              if(success):
                  ransacMatch = drawPairs(img1,img2,ransacPairs,path+str(i+1)+'_matching.png')
                  # cv2.imshow('ransacMatch',ransacMatch)


              """
              Image Warping + Blending
              Save Panorama output as mypano.png
              """
              if(not success): stitchedImg = img2
              else: stitchedImg = warpBlend(H,img1,img2,path+str(i+1)+'stitchedImg.png')
              cv2.imshow('stitchedImg',stitchedImg)


              cv2.imshow('img1',img1)
              cv2.imshow('img2',img2)
              if cv2.waitKey(0) & 0xFF==ord('q'):
                  break

    
if __name__ == '__main__':
    main()
 

