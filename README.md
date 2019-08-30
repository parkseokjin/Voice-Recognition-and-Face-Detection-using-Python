# Voice-Recognition-and-Face-Detection-using-Python

This project focuses on using Python-OpenCV for face recognition and Python Speech Recognition for voice recognition in order to create prototype program for image/voice based user recognition.


1. Python-OpenCV

In order to develop a reliable face recognition prototype program, we have implemented Python-OpenCV and its DNN APIs.

First, we have tested Python-OpenCV for face recognition with Haar Cascade API.

Although Haar Cascade is good at detecting human faces from direct line of sight.

However, even slight disruption in camera's line of sight dramatically decreases its detection capability.

In order to address this issue, we have implemented OpenCV DNN APIs for more robust face detection algorithm.

Thanks to pre-trained DNN networks, we were able to develop highly robust face detection algorithm that can detect faces even if the face is half-covered by obstacles.


2. Face Detection with DNN API and Scikit Learn SVM

For face detection, we have implemented Python-OpenCV DNN APIs and pre-trained face detection DNN network.

In order for the program to recognize the user based on the face images, we needed the classifier.

We used Python Scikit Learn SVM Classifier for classifying and labeling face images.

For SVM Classifier, we need to convert face images to numerical values. We applied pre-trained 128d embedder DNN network that converts face images to 128d matrix values.

During the training process, we used DNN network to detect faces from the image and convert them into 128d matrix values.

We train SVM classifier with 128 matrix values along with the names of faces/labels.

As a result, SVM Classifier can classify the face images according to the owners' names/labels.

During application, we use DNN network to detect faces from the image and convert them into 128d matrix values.

However, during application, instead of training, we give 128d matrix values to SVM Classifier and receive classified results, the names/labels of the face images.



3. Python Speech Recognition module & Google translate


