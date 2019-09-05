# Voice-Recognition-and-Face-Detection-using-Python

This project focuses on using Python-OpenCV for face recognition and Python Speech Recognition for voice recognition in order to create prototype program for image/voice based user recognition.


# 1. Python-OpenCV

In order to develop a reliable face recognition prototype program, we have implemented Python-OpenCV and its DNN APIs.

First, we have tested Python-OpenCV for face recognition with Haar Cascade API.

Although Haar Cascade is good at detecting human faces from direct line of sight, even the slightest disruption in camera's line of sight dramatically decreases its detection capability.

In order to address this issue, we have implemented OpenCV DNN APIs for more robust face detection algorithm.

Thanks to pre-trained DNN networks, we were able to develop highly robust face detection algorithm that can detect faces even if the face is half-covered by obstacles.


# 2. Face Detection with DNN API and Scikit Learn SVM

For face detection, we have implemented Python-OpenCV DNN APIs and pre-trained face detection DNN network.

In order for the program to recognize the user based on the face images, we needed the classifier.

We used Python Scikit Learn SVM Classifier for classifying and labeling face images.

For SVM Classifier, we need to convert face images to numerical values. We applied pre-trained 128d embedder DNN network that converts face images to 128d matrix values.

During the training process, we used DNN network to detect faces from the image and convert them into 128d matrix values.

We train SVM classifier with 128 matrix values along with the names of faces/labels.

As a result, SVM Classifier can classify the face images according to the owners' names/labels.

During application, we use DNN network to detect faces from the image and convert them into 128d matrix values.

However, during application, instead of training, we give 128d matrix values to SVM Classifier and receive classified results, the names/labels of the face images.

This process can be used to detect the user with face images detected from the camera.


# 3. Python Speech Recognition module (Google translate) & PyAudio

For voice recognition, we have implemented Python Speech Recognition module and PyAudio.

We use PyAudio in order to acquire real-time voice streaming data from microphone.

We directly apply voice streaming data from microphone into Python speech recognizer, which is linked with Google Translate API.

We can acquire STT (Speech-to-Text) results through Python speech recognizer and Google Translate API.

This process can create voice-based passwords for user recognition purposes.


# 4. Integration and Multi-Threading

Since face recognition and voice recognition have to run simultaneously, we needed to apply multi-threading.

We used ThreadPoolExecutor from thon concurrent module.

By assigning threads for each recognitioni process, we can run face recognition and voice recognition concurrently.
