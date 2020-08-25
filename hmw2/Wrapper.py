import cv2
import numpy as np
import glob
import scipy.optimize


def Vparams(H,i,j):
   vij = np.array([[H[0,i]*H[0,j]],
                   [H[0,i]*H[1,j] + H[1,i]*H[0,j]],
                   [H[1,i]*H[1,j]],
                   [H[2,i]*H[0,j] + H[0,i]*H[2,j]],
                   [H[2,i]*H[1,j] + H[1,i]*H[2,j]],
                   [H[2,i]*H[2,j]]])
   return vij


def Vmatrix(Hlist):
    V = np.zeros([1,6])
    for i in range(len(Hlist)):
        v01_T = Vparams(Hlist[i],0,1).T
        v00_T = Vparams(Hlist[i],0,0).T
        v11_T = Vparams(Hlist[i],1,1).T

        Vimg = np.vstack((v01_T,v00_T-v11_T)) 
        V  = np.vstack((V,Vimg))
    return V[1:,:]


def Bmatrix(V):
     U,S,V_T = np.linalg.svd(V)
     b = V_T[-1,:] 

     B = np.array([[b[0], b[1], b[3]],
                   [b[1], b[2], b[4]],
                   [b[3], b[4], b[5]]]) 
     return B


def getImgages():
   path = './Calibration_Imgs/*.jpg' 
   imgs = []
   for filename in glob.glob(path):
       img = cv2.imread(filename)
       img = cv2.resize(img,(500,500))
       imgs.append(img)
   return imgs


def extrinsicMatrix(A,H):
    Ainv = np.linalg.inv(A)

    #Rotation matrix approach 
    r0 = Ainv.dot(H[:,0])/((np.linalg.norm(Ainv.dot(H[:,0])) + np.linalg.norm(Ainv.dot(H[:,1])))/2)
    r1 = Ainv.dot(H[:,1])/((np.linalg.norm(Ainv.dot(H[:,0])) + np.linalg.norm(Ainv.dot(H[:,1])))/2)
    r2 = np.cross(r0,r1)
    t = Ainv.dot(H[:,2])/((np.linalg.norm(Ainv.dot(H[:,0])) + np.linalg.norm(Ainv.dot(H[:,1])))/2)
    t = t.reshape((3,1))#/t[-1]
    
    #Obtaining best Rotation matrix
    Q = np.column_stack((r0,r1,r2))
    U,S,V_T = np.linalg.svd(Q)
    R = np.dot(U,V_T)
    RT = np.hstack((R,t))
    return RT


def getCorrespondencies(imgs):
    cornerlist = []
    for img in imgs:
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,corners = cv2.findChessboardCorners(imgGray,(9,6),None)
        cornerlist.append(corners)

        # cv2.circle(img,(corners[0][0][0],corners[0][0][1]),5,[0,0,255])
        # cv2.circle(img,(corners[53][0][0],corners[53][0][1]),5,[0,0,255])

        # cv2.imshow('GrayImage', img)
        # if(cv2.waitKey(0) & 0XFF == ord('q')):
            # break
    return cornerlist


def projectionMatrices(cornerlist,Pw):
    Hlist = []
    for i in range(len(cornerlist)):
        H,_ = cv2.findHomography(Pw,cornerlist[i],cv2.RANSAC)
        Hlist.append(H)
    return Hlist


def errorFunc(params,cornerlist,Hlist,Pw):

   K = np.array([[params[0],         0,params[3]],
                 [        0, params[2],params[4]],
                 [        0,         0,       1]])

   k1, k2 = params[5], params[6] 
   
   zeros = np.zeros(len(Pw)).reshape((54,1))
   ones =  np.ones(len(Pw)).reshape((54,1))
   PwH = np.hstack((Pw,zeros,ones))
   u0,v0,_ = K[:,-1]
   
   error = 0

   for i in range(len(cornerlist)):
        RT = extrinsicMatrix(K,Hlist[i])
        x,y,z = RT@PwH.T
        x,y,z = x/z, y/z, z/z      
        
        u,v,z = K@RT@PwH.T
        u,v,z = u/z, v/z, z/z
         
        uh = (u + (u-u0)*(k1*(x**2 + y**2) + k2*(x**2+y**2)**2)).reshape((54,1))
        vh = (v + (v-v0)*(k1*(x**2 + y**2) + k2*(x**2+y**2)**2)).reshape((54,1))
        
         
        actualImgCoor = np.squeeze(np.array(cornerlist[i]))
        predictedImgCoor = np.column_stack((uh,vh)) 

        residual = np.linalg.norm(actualImgCoor-predictedImgCoor,axis=1).reshape((54,1))
        error = np.vstack((error,residual))

   error = np.squeeze(error)
   return error 
    

def main():
     imgs = getImgages()
      
     """
     Parameters
     """
     #checkboard parameters
     checkSz = 21.5; checkInnerRows = 6; checkInnerCols = 9
    
     #world coordinates
     r,c = np.mgrid[checkSz:(checkInnerRows+1)*checkSz:checkSz,
                    checkSz:(checkInnerCols+1)*checkSz:checkSz]
     Pw  = np.column_stack((c.ravel(),r.ravel()))

     #lens distortion initial estimate
     Kc = np.array([0,0])

     """
     Get corners for all the images
     """
     cornerlist = getCorrespondencies(imgs)

     """
     Obtain Projection matrix from point correspondencies
     """
     Hlist = projectionMatrices(cornerlist,Pw)
     
    
     """
     Find Vmatrix 
     """
     V = Vmatrix(Hlist)

     
     """
     Find symmetric B matrix
     """
     B = Bmatrix(V)


     """
     Find intrinsic K matrix
     """
     A = np.linalg.cholesky(B)
     K = np.linalg.inv(A.T)
     K = K/K[-1,-1]
    
     """
     Find K and Kc(lens distortion) of camera
     """
     #Assumption - assume skew s  = 0
     initParams = np.array([K[0,0],0,K[1,1],K[0,2],K[1,2],Kc[0],Kc[1]])
     res = scipy.optimize.least_squares(errorFunc,x0=initParams,method='lm',args=(cornerlist,Hlist,Pw))
     
     K = np.array([[res.x[0],res.x[1],res.x[3]],
                   [       0,res.x[2],res.x[4]],
                   [       0,       0,      1]])

     Kc = res.x[5:7]
     
     """
     Undistort image
     """
     params = np.array([Kc[0],Kc[1],0,0,0],dtype=float)
     
     for img in imgs:
         undist = cv2.undistort(img,K,params)

         cv2.imshow('undist', undist)
         cv2.imshow('img',img)
         if(cv2.waitKey(0) & 0XFF == ord('q')):
               break

if __name__ == '__main__':
    main()



