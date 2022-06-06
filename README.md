# Facial-Expression-Recognition.Pytorch
A CNN based pytorch implementation on facial expression recognition (FER2013 )

## Visualize for a test image by a pre-trained model ##
- python visualize.py

## FER2013 Dataset ##
labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

### Train and Eval model ###
- python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01

### plot confusion matrix ###
- python plot_fer2013_confusion_matrix.py --model VGG19 --split PrivateTest

### Train and Eval model for all 10 fold ###
- python k_fold_train.py


