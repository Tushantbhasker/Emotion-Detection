# Emotion-Detection
This repository is for project about emotion detection using convolutional neural network.

# Dataset
Dataset
We use the FER-2013 Faces Database, a set of 28,709 pictures of people displaying 7 emotional expressions (angry, disgusted, fearful, happy, sad, surprised and neutral). The dataset quality and image diversity is not very good and you will probably get a model with bad accuracy in other applications!

You have to request for access to the dataset or you can get it on Kaggle. Download fer2013.tar.gz and decompress fer2013.csv in the ./data folder.

# Building a CNN Model
* Read csv file
* Extract labels and pixel values of each image
* categorise dataset into training and test set
* Build a cnn model
* Fit model on training data
* Analyse and evaluate model on test data
* Save the weights of the model

# Using Open-cv for Live Emotion Detection
* Load our saved Model
* Use pretrained xml file to detect face
* resize face image for testing on model
* Predict output

# Dependencies
* Keras 2.2.4
* Tensorflow 1.1.10
* Numpy
* pandas
* Matplotlib
* Opencv
