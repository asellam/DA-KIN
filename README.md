# DA-KIN
Data Augmentation For Kinship  Verification

# Prerequisites
1. Python
1. Tensorflow (pip install tensorflow-gpu)
1. Numpy (pip install numpy)
1. Scipy (pip install scipy)
1. Keras 2.0

# How To Use
1. Run from Command Line Prompt: SiameseCNN.py (FoldNumber: 1..5) (Kinship Relation: fs, fd, ms, md) (Kinship Dataset: I, II) (Number of Data Augmentation Techniques) (Name of Technique 1) [Parameters  of Technique 1] (Name of Technique 2) [Parameters  of Technique 2] ...

1. Run using the batch file: SiameseCNN.bat, change the commands depending on your tests.

## Names of data augmentation techniques:
1. Flip: Image Horizontal Flipping
2. Crop: Image Cropping
3. Rotation: Image Rotation
4. Color: Image Color Channels Shifting
5. Light: Image Light Intensity Shifting

The results will be outputed to a csv file (scnn_results.csv)

# Important
Before running the code make sure to:
1. Download the KinFaceW data-sets from these links: http://www.kinfacew.com/dataset/KinFaceW-I.zip
http://www.kinfacew.com/dataset/KinFaceW-II.zip
1. Unzip the two files somewhere in your hard disk
1. Go to LoadData.py and set the constant RootDir to the path of the directory containing the two data sets: KinFaceW-I and KinFaceW-II
