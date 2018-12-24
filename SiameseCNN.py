#----------------------------------------#
#- Siamese CNN for Kinship Verification -#
#----------------------------------------#
#- By: Abdellah SELLAM                  -#
#-     Hamid AZZOUNE                    -#
#----------------------------------------#
#- Created: 2018-11-20                  -#
#- Last Update: 2018-12-24              -#
#----------------------------------------#

from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
import numpy.random as rng
import numpy as np
import sys
import csv
import os
#Our DataSet I/O Routines
from LoadData import LoadFold,GenerateSamples,SetDataParams,GetImageCrop,GetOutFileName
from matplotlib import pyplot as plt
from scipy import misc
from time import time

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)
T0=time()
#Parse command line arguments
F=int(sys.argv[1])
KinShip=sys.argv[2]
KinSet=sys.argv[3]

SetDataParams(sys.argv)

#Percentage of data used to validate
ValidSplit=0.0
LR=0.01
#Loads data of the Fold #F
#X0: Training Inputs
#Y0: Training Targets
#X2: Test Inputs
#Y2: Test Targets
(X0,Y0,X1,Y1)=LoadFold("KinFaceW-"+KinSet,KinShip,F,GetImageCrop())

#Inputs Array's Shape
in_shape=(X1.shape[2],X1.shape[3],X1.shape[4])
#Input Layer of the first (left) ConvNet (Images of parents)
left_input = Input(in_shape)
#Input Layer of the second (right) ConvNet (Images of children)
right_input = Input(in_shape)
#The definition of the convnet to be used for left and right inputs
convnet = Sequential()
#16 Convolutions of 13x13 size and 1x1 stride and ReLU activation
C1=Conv2D(16,(13,13),activation='relu',input_shape=in_shape,padding="same",
                kernel_regularizer=l2(0.001),kernel_initializer=W_init,bias_initializer=b_init)
convnet.add(C1)
#Max Pooling of 2x2 size and 2x2 stride
PL1=MaxPooling2D()
convnet.add(PL1)

#48 Convolutions of 5x5 size and 1x1 stride and ReLU activation
C2=Conv2D(48,(5,5),activation='relu',padding="same",
                kernel_regularizer=l2(0.001),kernel_initializer=W_init,bias_initializer=b_init)
convnet.add(C2)
#Max Pooling of 2x2 size and 2x2 stride
PL2=MaxPooling2D()
convnet.add(PL2)

#96 Convolutions of 3x3 size and 1x1 stride and ReLU activation
C3=Conv2D(96,(3,3),activation='relu',padding="same",
                kernel_regularizer=l2(0.001),kernel_initializer=W_init,bias_initializer=b_init)
convnet.add(C3)
#Max Pooling of 2x2 size and 2x2 stride
PL3=MaxPooling2D()
convnet.add(PL3)

#This layer transform a 3D volume to a 1D vector by a flattenning operation
convnet.add(Flatten())
#Left input (Parent Image) encoded by the convnet into a 1D feature vector
encoded_l = convnet(left_input)
#Right input (Child Image) encoded by the convnet into a 1D feature vector
encoded_r = convnet(right_input)
#Define the L1 lambda function to be used in merging encoded inputs
L1_distance = lambda x: K.abs(x[0]-x[1])
#Merge the two encoded inputs (1D feature vectors) using L1 norm
both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
# A dense layer with Sigmoid activation applied to the merged vectors to
# reduce features
Features = Dense(128,activation='sigmoid',bias_initializer=b_init)(both)
# A final dense layer with Sigmoid activation applied to compute kinship
# output
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(Features)
#The Siamese CNN defined as model taking as input two facial images and
# outputting the kinship output computed by applying the final dense layer
# to the merged vectors extracted by the two identical convnets sharing the
# same set of weights!
SiameseCNN = Model(input=[left_input,right_input],output=prediction)
#Compile the Keras Model
SiameseCNN.compile(loss="mean_squared_error",optimizer=SGD(LR),metrics=["accuracy"])
#print the number of Training/Validation samples
#Training the Siamese CNN with a goal loss
#Current Epoch Counter
Epochs=0
#Maximum neumber of epochs
MaxEpochs=150
#Current accuracy on training data
TrainAccu=0.0
#Current accuracy on validation data
ValidAccu=0.0
BatchSize=12000
#Conitnue as long as not reached maximum number of epochs and one of the loss
# values (for train/validation data) did not reach its goal value
BestTestAccu=0.0
BestValidAccu=0.0
BestTrainAccu=0.0
BestEpoch=0
print("Training on",BatchSize,"samples, validation on",X1.shape[0],"samples")
while (Epochs<MaxEpochs)and(TrainAccu<0.9999):
    startD=time()
    #A single epoch training
    (X3,Y3)=GenerateSamples(X0,Y0,BatchSize)
    startT=time()
    Hist=SiameseCNN.fit(X3,Y3,validation_data=([X1[:,0,:,:,:],X1[:,1,:,:,:]],Y1),epochs=1,verbose=0,batch_size=200)
    #Compute Training Data's loss/accuracy
    #Training Data's accuracy
    TrainAccu=Hist.history["acc"][0]
    #Validation Data's accuracy
    ValidAccu=Hist.history["val_acc"][0]
    #Increment the epoch's counter
    Epochs=Epochs+1
    if(ValidAccu>BestValidAccu)and(TrainAccu>ValidAccu):
        SiameseCNN.save("./Models/SCNN_TEMP.h5")
        TestE=SiameseCNN.evaluate(x=[X1[:,0,:,:,:],X1[:,1,:,:,:]],y=Y1,verbose=0)
        BestValidAccu=ValidAccu
        BestTrainAccu=TrainAccu
        BestTestAccu=TestE[1]
        BestEpoch=Epochs
    #Display Epoch, loss, accuracy ...
    print("Epoch: %d/%d"%(Epochs,MaxEpochs))
    print("loss: %.04f - accuracy: %.04f - valid_loss: %.04f - valid_accuracy: %.04f"%(Hist.history["loss"][0],Hist.history["acc"][0],Hist.history["val_loss"][0],Hist.history["val_acc"][0]))
    print("%ds+%ds"%(startT-startD,int(time()-startT)))


#Save accuracies for this Fold as a Comma-Separated-Values file
#Read old accuracies if they exists
FileName="./Results/Results%s.csv"%(GetOutFileName())
if(os.path.isfile(FileName)):
    csvr=open(FileName,"r")
    rows=csv.reader(csvr,delimiter=';')
    csvd=[row for row in rows]
    data={"fs-I":[csvd[i][1] for i in range(1,6)],
          "fd-I":[csvd[i][2] for i in range(1,6)],
          "ms-I":[csvd[i][3] for i in range(1,6)],
          "md-I":[csvd[i][4] for i in range(1,6)],
          "fs-II":[csvd[i][5] for i in range(1,6)],
          "fd-II":[csvd[i][6] for i in range(1,6)],
          "ms-II":[csvd[i][7] for i in range(1,6)],
          "md-II":[csvd[i][8] for i in range(1,6)]}
    csvr.close()
else:
    data={"fs-I":["0.00","0.00","0.00","0.00","0.00"],"fd-I":["0.00","0.00","0.00","0.00","0.00"],
          "ms-I":["0.00","0.00","0.00","0.00","0.00"],"md-I":["0.00","0.00","0.00","0.00","0.00"],
          "fs-II":["0.00","0.00","0.00","0.00","0.00"],"fd-II":["0.00","0.00","0.00","0.00","0.00"],
          "ms-II":["0.00","0.00","0.00","0.00","0.00"],"md-II":["0.00","0.00","0.00","0.00","0.00"]}

KinKey=KinShip+"-"+KinSet
TestE=SiameseCNN.evaluate(x=[X1[:,0,:,:,:],X1[:,1,:,:,:]],y=Y1,verbose=0)
# if new results are better then save this model
if BestTestAccu>float(data[KinKey][F-1]):
    ModelName="./Models/SCNN_%s_%s_%d%s.h5"%(KinShip,KinSet,F,GetOutFileName())
    data[KinKey][F-1]="%.04f"%(BestTestAccu)
    if os.path.isfile("./Models/SCNN_TEMP.h5"):
        if os.path.isfile(ModelName):
            os.remove(ModelName)
        os.rename("./Models/SCNN_TEMP.h5",ModelName)

# Display Results
print("Best Epoch: %d\n\t Train Accuracy: %.04f\n\t Validation Accuracy: %.04f\n"%(BestEpoch,BestTrainAccu,BestValidAccu))
print("Fold:",F)
print("DataSet:",KinKey)
print("Accuracy:",BestTestAccu)

# Write saved data to disc as a Comma-Separated-Values file
csvw=open(FileName,"w")
csvw.write("Fold;fs-I;fd-I;ms-I;md-I;fs-II;fd-II;ms-II;md-II\n")
for F in range(5):
    csvw.write("%d;%s;%s;%s;%s;%s;%s;%s;%s\n"%(F+1,data["fs-I"][F],data["fd-I"][F],data["ms-I"][F],data["md-I"][F],data["fs-II"][F],data["fd-II"][F],data["ms-II"][F],data["md-II"][F]))
csvw.close()
T1=time()
print("Total Time: %.01fs"%(T1-T0))
