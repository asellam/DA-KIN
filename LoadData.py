#----------------------------------------#
#- Siamese CNN for Kinship Verification -#
#- Kinship data manipulation routines   -#
#----------------------------------------#
#- By: Abdellah SELLAM                  -#
#-     Hamid AZZOUNE                    -#
#----------------------------------------#
#- Created: 2018-11-20                  -#
#- Last Update: 2018-12-24              -#
#----------------------------------------#

import numpy as np
import scipy.ndimage as nd
import scipy.io as sio
from scipy import misc
from matplotlib import pyplot as plt
from random import randint,randrange,uniform
# Change this value to the path of the directory (folder) containing
# the KinFaceW-I and KinfaceW-II folders
RootDir="D:/PhD"
# A dictionnary to converts the kinship relation prefix to the name of
# directory containing the images
PrefixToDir={"fd":"father-dau","fs":"father-son","md":"mother-dau","ms":"mother-son"}

# Constants identifiying the techniques of data augmentation
NoneOp=0
CropOp=1
FlipOp=2
RotationOp=3
ColorOp=4
LightOp=5
WidthOp=6
HeightOp=7

# Flags indicating the use or not of each data augmentation technique
CropF=False
FlipF=False
RotationF=False
ColorF=False
LightF=False
WidthF=False
HeightF=False
# Data augmentation parameters (ranges)
CropR=0
RotationR=0
ColorR=0.0
LightR=0.0
WidthR=0
HeightR=0

# A function that horizontally flips the input image and returns the
# flipped version
def FlipImg(img):
    return np.fliplr(img)

# A function that rotates the input image with an input angle and 
# returns the rotated one
def RotateImg(img,ang):
    theta=np.pi/180*ang
    c=np.cos(theta)
    s=np.sin(theta)
    x=img.shape[1]/2+0.5
    y=img.shape[0]/2+0.5
    R=img[:,:,0]
    G=img[:,:,1]
    B=img[:,:,2]
    rotation_matrix = np.array([[+c,-s,+0],
                                [+s,+c,+0],
                                [+0,+0,+1]])
    offset_matrix = np.array([[+1,+0,+y],
                              [+0,+1,+x],
                              [+0,+0,+1]])
    reset_matrix = np.array([[+1,+0,-y],
                             [+0,+1,-x],
                             [+0,+0,+1]])
    transofrm_matrix= np.dot(np.dot(offset_matrix, rotation_matrix), reset_matrix)
    RR=nd.affine_transform(R,transofrm_matrix,mode='nearest')
    RG=nd.affine_transform(G,transofrm_matrix,mode='nearest')
    RB=nd.affine_transform(B,transofrm_matrix,mode='nearest')
    return np.stack([RR,RG,RB],axis=-1)

# A function that multiplies the channels of the input image by real
# numbers to shift them
def ShiftImg(img,shift):
    return img*shift 

# A function to resize the the input image
def ResizeImg(img,ShiftX,ShiftY):
    R=img[:,:,0]
    G=img[:,:,1]
    B=img[:,:,2]
    x=ShiftX/2
    y=ShiftY/2
    s=img.shape[1]/(img.shape[1]-ShiftX)
    t=img.shape[0]/(img.shape[0]-ShiftY)
    transofrm_matrix = np.array([[+t,+0,-y],
                                 [+0,+s,-x],
                                 [+0,+0,+1]])
    RR=nd.affine_transform(R,transofrm_matrix,mode='nearest')
    RG=nd.affine_transform(G,transofrm_matrix,mode='nearest')
    RB=nd.affine_transform(B,transofrm_matrix,mode='nearest')
    return np.stack([RR,RG,RB],axis=-1)

#This function loads data from a single kinship subset
#Arguments:
#----------
#KinSet: The kinship dataset version (KinFaceW-I or KinFaceW-II)
#KinShip: The kinship subset prefix (fd,fs,md or ms)
#Fold: The Five-Fold-Cross-Validation's fold number (1,2,3,4 or 5)
#ValidSplit: The proportion of training data to be used for validation ([0..1])
#Return Value:
#--------------
#(X0,Y0,X1,Y1): a tuple containing Training/Validation/Test inputs and
# targets:
#   X0: Training Inputs (Numpy nd-array)
#   Y0: Training Targets (Numpy nd-array)
#   X1: Test Inputs (Numpy nd-array)
#   Y1: Test Targets (Numpy nd-array)
def LoadFold(KinSet,KinShip,Fold,CropR):
    meta=sio.loadmat(RootDir+"/"+KinSet+"/meta_data/"+KinShip+"_pairs.mat")
    pairs=meta['pairs']
    TrainX=[]
    TrainY=[]
    ValidX=[]
    ValidY=[]
    TestX=[]
    TestY=[]
    pDir=RootDir+"/"+KinSet+"/images/"+PrefixToDir[KinShip]+"/"
    for p in pairs:
        pImg=misc.imread(pDir+p[2][0])
        cImg=misc.imread(pDir+p[3][0])
        if p[0][0][0]==Fold:
            pImgC=pImg[CropR:64-CropR,CropR:64-CropR,:]
            cImgC=cImg[CropR:64-CropR,CropR:64-CropR,:]
            TestX.append([pImgC,cImgC])
            TestY.append([p[1][0][0]])
        else:
            TrainX.append([pImg,cImg])
            TrainY.append([p[1][0][0]])
    return (np.array(TrainX),np.array(TrainY),np.array(TestX),np.array(TestY))

# This function returns the current amount of image cropping used
def GetImageCrop():
    global CropF
    global CropR
    if(CropF):
        return CropR
    return 0

# This generates a file name based on the techniques used in data
# augmentation, this file name is used to output results and models
def GetOutFileName():
    global CropF,FlipF,RotationF,ColorF,LightF,WidthF,HeightF
    global CropR,RotationR,ColorR,LightR,WidthR,HeighR
    FileName=""
    if(CropF):
        FileName=FileName+"_P_%d"%(CropR)
    if(FlipF):
        FileName=FileName+"_F"
    if(RotationF):
        FileName=FileName+"_R_%d"%(RotationR)
    if(ColorF):
        FileName=FileName+"_C_%g"%(ColorR)
    if(LightF):
        FileName=FileName+"_L_%g"%(LightR)
    if(WidthF):
        FileName=FileName+"_W_%d"%(WidthR)
    if(HeightF):
        FileName=FileName+"_H_%d"%(HeightR)
    return FileName

# This function converts commnad line arguments into data augmentation 
# parameters
# The command line must contain the foloowing items:
# 1. Fold Number {1, 2, 3, 4, 5}
# 2. Kinship relation:
#    - fs: father-son
#    - fd: father-daughter
#    - ms: mother-son
#    - md: mother-daughter
# 3. Kinship dataset version:
#    -  I: KinFaceW-I
#    - II: KinFaceW-II
# 4. The number of data augmentation techniques to be used
# 5. Techniques to be used and their parameters
# Example:
#   SiameseCNN.py 3 ms II 3 Flip Crop 8 Rotation 15
# Names of data augmentation techniques:
#   1. Flip: Image Horizontal Flipping
#   2. Crop: Image Cropping
#   3. Rotation: Image Rotation
#   4. Color: Image Color Channels Shifting
#   5. Light: Image Light Intensity Shifting
def SetDataParams(argv):
    global CropF,FlipF,RotationF,ColorF,LightF,WidthF,HeightF
    global CropR,RotationR,ColorR,LightR,WidthR,HeighR

    N=int(argv[4])
    O=5
    for i in range(N):
        if(argv[O]=="Crop"):
            CropF=True
            CropR=int(argv[O+1])
            O=O+2
        else:
            if(argv[O]=="Flip"):
                FlipF=True
                O=O+1
            else:
                if(argv[O]=="Rotation"):
                    RotationF=True
                    RotationR=int(argv[O+1])
                    O=O+2
                else:
                    if(argv[O]=="Color"):
                        ColorF=True
                        ColorR=float(argv[O+1])
                        O=O+2
                    else:
                        if(argv[O]=="Light"):
                            LightF=True
                            LightR=float(argv[O+1])
                            O=O+2
                        else:
                            if(argv[O]=="Width"):
                                WidthF=True
                                WidthR=int(argv[O+1])
                                O=O+2
                            else:
                                if(argv[O]=="Height"):
                                    HeightF=True
                                    HeightR=int(argv[O+1])
                                    O=O+2

# This function is called before each training epoch to generate
# new augmented training samples
def GenerateSamples(X,Y,BatchSize):
    SamplesP=[]
    SamplesC=[]
    SamplesK=[]
    Operations=[]
    if(CropF):
        Operations.append(CropOp)
    if(FlipF):
        Operations.append(FlipOp)
    if(RotationF):
        Operations.append(RotationOp)
    if(ColorF):
        Operations.append(ColorOp)
    if(LightF):
        Operations.append(LightOp)
    if(WidthF):
        Operations.append(WidthOp)
    if(HeightF):
        Operations.append(HeightOp)
    nOperation=len(Operations)
    if nOperation==0:
        return ([X[:,0,:,:,:],X[:,1,:,:,:]],Y)
    CropP=GetImageCrop()
    sp=1
    for i in range(BatchSize):
        #Next pair's images
        pImg0=X[sp][0]
        cImg0=X[sp][1]
        OP=randint(0,nOperation-1)
        if Operations[OP]==CropOp:
            MCrop=CropR+CropR
            SCrop=64-MCrop
            PCropX=randint(0,MCrop)
            PCropY=randint(0,MCrop)
            CCropX=randint(0,MCrop)
            CCropY=randint(0,MCrop)
            pImg=pImg0[PCropY:PCropY+SCrop,PCropX:PCropX+SCrop,:]
            cImg=cImg0[CCropY:CCropY+SCrop,CCropX:CCropX+SCrop,:]
        if Operations[OP]==FlipOp:
            FlipP=randint(0,1)
            FlipC=randint(0,1)
            if(FlipP==1):
                pImg1=FlipImg(pImg0)
            else:
                pImg1=pImg0
            if(FlipC==1):
                cImg1=FlipImg(cImg0)
            else:
                cImg1=cImg0
            pImg=pImg1[CropP:64-CropP,CropP:64-CropP,:]
            cImg=cImg1[CropP:64-CropP,CropP:64-CropP,:]
        if Operations[OP]==RotationOp:
            RotationP=randint(-RotationR,+RotationR)
            RotationC=randint(-RotationR,+RotationR)
            pImg1=RotateImg(pImg0,RotationP)
            cImg1=RotateImg(cImg0,RotationC)
            pImg=pImg1[CropP:64-CropP,CropP:64-CropP,:]
            cImg=cImg1[CropP:64-CropP,CropP:64-CropP,:]
        if Operations[OP]==ColorOp:
            MinC=1.0-ColorR
            MaxC=1.0+ColorR
            ColorP=[uniform(MinC,MaxC),uniform(MinC,MaxC),uniform(MinC,MaxC)]
            ColorC=[uniform(MinC,MaxC),uniform(MinC,MaxC),uniform(MinC,MaxC)]
            pImg1=ShiftImg(pImg0,ColorP)
            cImg1=ShiftImg(cImg0,ColorC)
            pImg=pImg1[CropP:64-CropP,CropP:64-CropP,:]
            cImg=cImg1[CropP:64-CropP,CropP:64-CropP,:]
        if Operations[OP]==LightOp:
            MinL=1.0-LightR
            MaxL=1.0+LightR
            LightPV=uniform(MinL,MaxL)
            LightCV=uniform(MinL,MaxL)
            LightP=[LightPV,LightPV,LightPV]
            LightC=[LightCV,LightCV,LightCV]
            pImg1=ShiftImg(pImg0,LightP)
            cImg1=ShiftImg(cImg0,LightC)
            pImg=pImg1[CropP:64-CropP,CropP:64-CropP,:]
            cImg=cImg1[CropP:64-CropP,CropP:64-CropP,:]
        if Operations[OP]==WidthOp:
            WidthP=randint(0,WidthR)
            WidthC=randint(0,WidthR)
            pImg1=ResizeImg(pImg0,WidthP,0)
            cImg1=ResizeImg(cImg0,WidthC,0)
            pImg=pImg1[CropP:64-CropP,CropP:64-CropP,:]
            cImg=cImg1[CropP:64-CropP,CropP:64-CropP,:]
        if Operations[OP]==HeightOp:
            HeightP=randint(0,HeightR)
            HeightC=randint(0,HeightR)
            pImg1=ResizeImg(pImg0,0,HeightP)
            cImg1=ResizeImg(cImg0,0,HeightC)
            pImg=pImg1[CropP:64-CropP,CropP:64-CropP,:]
            cImg=cImg1[CropP:64-CropP,CropP:64-CropP,:]
        SamplesP.append(pImg)
        SamplesC.append(cImg)
        SamplesK.append(Y[sp])
        sp=sp+1
        if sp==X.shape[0]:
            sp=0
    return ([np.array(SamplesP),np.array(SamplesC)],np.array(SamplesK))