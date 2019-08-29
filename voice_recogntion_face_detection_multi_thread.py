import numpy as np
import argparse
import pickle
import os
import sys


sys.path.insert(0,'/usr/local/lib/python2.7/dist-packages')
print('OpenCv-Python Path : ' + sys.path[0])
import cv2

from concurrent.futures import ThreadPoolExecutor
import speech_recognition as sr
from os import path 
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import speech_recognition as sr
from os import path
import pyaudio
from six.moves import queue

import wave

import time

face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml')

def face_detection():
	mode = input("input mode num: ")
	if(mode == 0):
		img = cv2.imread('image/img.jpg') 
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		if(img is None):
			print('Empty img')
		for (x,y,w,h) in faces:
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	
		cv2.imshow('img',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	elif(mode == 1):

		capture = cv2.VideoCapture(0)
		capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)	
		if(capture.isOpened()):
			print('video cam is available')
			while(capture.isOpened()):
				ret, frame = capture.read()
	
				if(ret==True):
					gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
					faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
					print(faces)
	
					for (x,y,w,h) in faces:
						frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
						roi_gray = gray[y:y+h, x:x+w]
						roi_color = frame[y:y+h, x:x+w]
					
					cv2.imshow('image', frame)
        
				if(cv2.waitKey(1) > 0): break

			capture.release()
			cv2.destroyAllWindows()
	elif(mode == 2):
		print('Face Detection for a static image using OpenCV DNN module')
		print('Loading Neural Net')
	
		net = cv2.dnn.readNetFromCaffe('face_detection/DNN_module/deploy.prototxt.txt',
			'face_detection/DNN_module/res10_300x300_ssd_iter_140000.caffemodel')

		if(net is None):
			print('Failed to load Neural Net')
			sys.exit(0)
		elif(net is not None):
			print('Neural Net ready')
			img = cv2.imread('face_detection/image/test_person.png')
			(h, w) = img.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
			net.setInput(blob)
			detections = net.forward()
			compare_confidence_val = 0.5

			for i in range(0, detections.shape[2]):

				confidence = detections[0, 0, i, 2]

				if confidence > compare_confidence_val:

					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					text = "{:.2f}%".format(confidence * 100)
					y = startY - 10 if startY - 10 > 10 else startY + 10
					cv2.rectangle(img, (startX, startY), (endX, endY), (0,0,255), 2)
					cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)

		
			cv2.imshow("DNN Output", img)
			cv2.waitKey(0)
	elif(mode == 3):
		print('Face detection for video images using OpenCV DNN Module')
		protopath = 'face_detection/DNN_module/deploy.prototxt.txt'
		modelpath = 'face_detection/DNN_module/res10_300x300_ssd_iter_140000.caffemodel'

		capture = cv2.VideoCapture(0)

		if(capture.isOpened()):
			print('Video cam is available')
			capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
			capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
			net = cv2.dnn.readNetFromCaffe(protopath, modelpath)

			if(net is None): 
				print('Failed to load Neural Net') 
				sys.exit(0)
			elif(net is not None):

				print('Neural Net ready')
				while(capture.isOpened()):
					ret, frame = capture.read()

					if(ret == True):
					
						(h, w) = frame.shape[:2]
						blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
						net.setInput(blob)
						detections = net.forward()
						compare_confidence_val = 0.5

						for i in range(0, detections.shape[2]):
							confidence = detections[0, 0, i, 2]
				
							if confidence > compare_confidence_val:
								box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
								(startX, startY ,endX, endY) = box.astype("int")
								text = "{:.2f}%".format(confidence * 100)
								y = startY - 10 if startY - 10 > 10 else startY + 10
								cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
								cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
					
						cv2.imshow('Video Frame', frame)
						if(cv2.waitKey(1) > 0):break
				capture.release()
				cv2.destroyAllWindows()

		else: 
			print('Failed to opend video feed') 
			sys.exit(0)

	elif(mode == 4):

		protopath = 'face_detection/DNN_module/deploy.prototxt.txt'
		detector_modelpath = 'face_detection/DNN_module/res10_300x300_ssd_iter_140000.caffemodel'
		embedder_modelpath = 'face_detection/DNN_module/OpenFace.nn4.small2.v1.t7'
		dataset_path = 'face_detection/dataset' 

		detector = cv2.dnn.readNetFromCaffe(protopath, detector_modelpath)
		if(detector is None):
			print('Failed to load Detector Neural Net')
			sys.exit(0)
		elif(detector is not None):
			print('Detector Neural Net ready')

		embedder = cv2.dnn.readNetFromTorch(embedder_modelpath)
		if(embedder is None):
			print('Failed to load Embedder Neural Net')
			sys.exit(0)
		elif(detector is not None):
			print('Embedder Neural Net ready')

		knownEmbeddings = []
		knownNames = []

		total = 0

		compare_confidence_val = 0.3
		train_image_width = 600
		train_image_height = 0

		trainPaths = list(os.listdir(dataset_path))
		imagePaths = []
		for i in range(len(trainPaths)):
			imagePaths.append(os.listdir(dataset_path + '/' + trainPaths[i]))
		for(i, imagePath) in enumerate(imagePaths):
			name =  trainPaths[i]
			print('Embedding target : ' + name)
			for j in range(len(imagePath)):
			
				readPath = dataset_path + '/' + trainPaths[i] + '/' + imagePath[j]
				print('Reading : ' + readPath)

				image = cv2.imread(readPath)
				(h, w) = image.shape[:2]

				train_image_height = int((train_image_width * h)//w)
				image = cv2.resize(image, (train_image_width, train_image_height))
				(h, w) = image.shape[:2]

				imageblob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
				detector.setInput(imageblob)
				detections = detector.forward()		
	 
				if len(detections) > 0:

					k = np.argmax(detections[0, 0, :, 2])
					confidence = detections[0, 0, k, 2]

					if confidence > compare_confidence_val:

						box = detections[0, 0, k, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")

						face = image[startY:endY, startX:endX]
						(fH, fW) = face.shape[:2]

						if fW < 20 or fH < 20:
							continue
						faceblob =  cv2.dnn.blobFromImage(face, 1.0/255, (96,96),(0,0,0), swapRB=True, crop = False)
						embedder.setInput(faceblob)
						vec = embedder.forward()
						knownNames.append(name)
						knownEmbeddings.append(vec.flatten())
						total += 1

						print('[INFO] Serializing {} encodings...'.format(total))
						data = {"embeddings" : knownEmbeddings, "names" : knownNames}
						f = open('face_detection/output/embeddings.pickle', 'wb')
						f.write(pickle.dumps(data))
						f.close()

		print('[INFO] Loading face embeddings')
		data = pickle.loads(open('face_detection/output/embeddings.pickle','rb').read())

		print('[INFO] Encoding lables')
		le = LabelEncoder()
		labels = le.fit_transform(data['names'])

		print('[INFO] Training model')
		recognizer = SVC(C=1.0, kernel='linear', probability=True)
		recognizer.fit(data['embeddings'], labels)

		f = open('face_detection/output/le.pickle', 'wb')
		f.write(pickle.dumps(le))
		f.close()

		f = open('face_detection/output/recognizer.pickle', 'wb')
		f.write(pickle.dumps(recognizer))
		f.close()


		protopath = 'face_detection/DNN_module/deploy.prototxt.txt'
		modelpath = 'face_detection/DNN_module/res10_300x300_ssd_iter_140000.caffemodel'
		detector_modelpath = 'face_detection/DNN_module/res10_300x300_ssd_iter_140000.caffemodel'
		embedder_modelpath = 'face_detection/DNN_module/OpenFace.nn4.small2.v1.t7'

		capture = cv2.VideoCapture(0)
	
		detector = cv2.dnn.readNetFromCaffe(protopath, detector_modelpath)
		if(detector is None):
			print('Failed to load Detector Neural Net')
			sys.exit(0)
		elif(detector is not None):
			print('Detector Neural Net ready')

		embedder = cv2.dnn.readNetFromTorch(embedder_modelpath)
		if(embedder is None):
			print('Failed to load Embedder Neural Net')
			sys.exit(0)
		elif(detector is not None):
			print('Embedder Neural Net ready')
  
		recognizer = pickle.loads(open('face_detection/output/recognizer.pickle','rb').read())
		le = pickle.loads(open('face_detection/output/le.pickle','rb').read())
	
		if(capture.isOpened()):
			print('Video cam is available')
			capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
			capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
			net = cv2.dnn.readNetFromCaffe(protopath, modelpath)
			if(net is None): 
				print('Failed to load Neural Net') 
				sys.exit(0)
			elif(net is not None):
				print('READY')
				while(capture.isOpened()):
					ret, frame = capture.read()
					if(ret == True):

						target_frame_path = 'frame'
						recognizer = pickle.loads(open('face_detection/output/recognizer.pickle','rb').read())
						le = pickle.loads(open('face_detection/output/le.pickle','rb').read())

						train_image_width = 600
						train_image_height = 0

						compare_confidence_val = 0.3

						video = cv2.imread(target_frame_path)
						(h, w) = frame.shape[:2]

						train_image_height = int((train_image_width * h)//w)
						frame = cv2.resize(frame, (train_image_width, train_image_height))
						(h, w) = frame.shape[:2]

						imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

						detector.setInput(imageBlob)
						detections = detector.forward()
	
						for i in range(0, detections.shape[2]):
							confidence = detections[0, 0, i, 2]

							if (confidence > compare_confidence_val):
								box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
								(startX, startY, endX, endY) = box.astype('int')
								face = image[startY:endY, startX:endX]
								(fH, fW) = face.shape[:2]

								if fW < 20 or fH < 20:
									continue

								faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0,0,0), swapRB=True, crop=False)
								embedder.setInput(faceBlob)
								vec = embedder.forward()

								predictions = recognizer.predict_proba(vec)[0]
								j = np.argmax(predictions)
								probability = predictions[j]
								name = le.classes_[j]
			
							#for a in range(len(predictions)):
							#	pred_text = "{} : {:.2f}%".format(le.classes_[a], predictions[a] * 100)
							#	print(pred_text)
								if(probability>=0.5):

									text = "{} : {:.2f}%".format(name, probability * 100)
									y = startY - 10 if startY - 10 > 10 else startY + 10

									cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
									cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
								else:
									print('WRONG FACE')
						cv2.imshow('Output', frame)
						if(cv2.waitKey(1) > 0):break

				capture.release()
				cv2.destroyAllWindows()


def voice_recognition():
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 44100
	RECORD_SECONDS = 3
	WAVE_OUTPUT_FILENAME = "output.wav"

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
    	            channels=CHANNELS,
        	        rate=RATE,
            	    input=True,
    	            frames_per_buffer=CHUNK)

	print("Start to record the audio.")

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    	data = stream.read(CHUNK)
    		frames.append(data)

	print("Recording is finished.")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

	AUDIO_FILE = path.join(path.dirname(path.realpath('/home/luwis/Workspace/face_detection_ML_DNN/output.wav')),"output.wav") 
	
	print("Recorded audio path : " + AUDIO_FILE)

	r = sr.Recognizer()
	with sr.AudioFile(AUDIO_FILE) as source: 
		audio = r.record(source) 

		try:
			print("Google Speech Recognition thinks you said" + r.recognize_google(audio, language="ko-KR")+ "in korean") 
		except sr.UnknownValueError: 
			print("Google Speech Recognition could not understand audio")
		except sr.RequestError as e: 
			print("Could nit request results from Google Speech Recognition service; {0}".format(e))


pool_executor = ThreadPoolExecutor(1)

pool = pool_executor.submit(face_detection)

if pool.done():
	print("pool result : " + pool.result())
else:
	print('not yet')

pool2_executor = ThreadPoolExecutor(2)

pool2 = pool2_executor.submit(voice_recognition)

if pool2.done():
	print("pool2 result : " + pool2.result())
else:
	print('NOT YET')
