# Facial-Expression-Recognition.Pytorch
A CNN based pytorch implementation on facial expression recognition (FER2013 ), achieving 73.112% (state-of-the-art) in FER2013 


## Dependencies ##
- Python 2.7
- Pytorch >=0.2.0
- h5py (Preprocessing)
- sklearn (plot confusion matrix)

## Visualize for a test image by a pre-trained model ##
- python visualize.py

## FER2013 Dataset ##
- Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Image Properties: 48 x 48 pixels (2304 bytes)
labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

### Preprocessing Fer2013 ###
- first download the dataset(fer2013.csv) then put it in the "data" folder, then
- python preprocess_fer2013.py

### Train and Eval model ###
- python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01

### plot confusion matrix ###
- python plot_fer2013_confusion_matrix.py --model VGG19 --split PrivateTest

###              fer2013 Accurary             ###

- Model：    VGG19 ;       PublicTest_acc：  71.496% ;     PrivateTest_acc：73.112%     <Br/>
- Model：   Resnet18 ;     PublicTest_acc：  71.190% ;    PrivateTest_acc：72.973%     


### Train and Eval model for all 10 fold ###
- python k_fold_train.py


