import numpy as np
import argparse #인자를 입력받고,예외처리,사용법 작성등을 자동으로 해주는 모듈 argparse.
import pickle #리스트 등의 자료 형태를 자료형의 변경없이 파일로 저장하기 위해 pickle모듈 사용.
import os #특정 디렉터리 내의 파일 목록을 구하고자 할때 필요한 os모듈.
import sys
sys.path.append('/usr/local/python/cv2/python-2.7')
import cv2

from sklearn.preprocessing import LabelEncoder # 분류화된 Feature들을 수치적 값에 대해 Label Naming을 위한 sklearn labelencoder
from sklearn.svm import SVC # 여러가지 데이터들에 대해 Feature에 따라 분류하기 위해 ML SVM 기반 Classifier

face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml')


mode = 4

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
		#저장되어있던 네트워크 모델을 읽는다
	if(net is None):
		print('Failed to load Neural Net')
		sys.exit(0)
	elif(net is not None):
		print('Neural Net ready')
		img = cv2.imread('face_detection/image/test_person.png')
		(h, w) = img.shape[:2]#슬라이싱과 시퀀스 언패킹의 조합,리스트에서 첫 두요소만(높이 너비)
				       잡을 수 있다.
		blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))#이미지에서 사차원형의 블롭을 형성한다.
		net.setInput(blob) #네트워크에서 blob이라는 새로운 입력값을 설정한다.
		detections = net.forward() #출력 레이어값을 계속 계산해 나간다.
		compare_confidence_val = 0.5

		for i in range(0, detections.shape[2]):

			confidence = detections[0, 0, i, 2]#약한 detection을 확

			if confidence > compare_confidence_val:

				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				text = "{:.2f}%".format(confidence * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				# 요소들이 프레임이나 차원 밖으로 나가지 않도록 보정해준다
				cv2.rectangle(img, (startX, startY), (endX, endY), (0,0,255), 2)
				cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)#신뢰도가 임계값 이상인 사각형을 그린다.text문자열을 통해 확률을 표현한다.
 		   #rectangle과 puttext를 이용해 경계 박스와 텍스트를 그린다. 

		
		cv2.imshow("DNN Output", img)
		cv2.waitKey(0)

elif(mode == 3):

	print('Face detection for video images using OpenCV DNN Module')
	protopath = 'face_detection/DNN_module/deploy.prototxt.txt'
	#protopath: 프로토텍스트 (Neural Network Node별 설정을 저장한 텍스트 파일) 파일이 저장된 경로	
	modelpath = 'face_detection/DNN_module/res10_300x300_ssd_iter_140000.caffemodel'    
	#modetpath:학습된 네트워크가 정의되어 있는 파일이 저장된 경로

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
	# readnetfromtorch는 pyTorch로 제작된 Neural Network 모델을 읽어주는 함수임
 
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
	# dataset 파일에 있는 이미지를 저장하고 있는 파일 이름을 모두 리스트형으로 가져온다

	imagePaths = []
	for i in range(len(trainPaths)):
		imagePaths.append(os.listdir(dataset_path + '/' + trainPaths[i]))
	 	#imagepath리스트에 dataset_path와 trainpath를 짝지어서 넣어준다.
		for(i, imagePath) in enumerate(imagePaths):
		# in enumerate(imagePaths) : i번째 index와 i번째 index에 해당되는 imagePaths 리스트에 저장된 값을 (i, imagePath)로 짝지어서 각각 i와 imagePath에 저장함

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
				# cv2.dnn.blobFromImage : 4차원형의 blob을 얻기위한 작업.
				# [주요 인자]
				# input image : 입력 이미지 데이터
				# scalefactor : mean substraction시 사용할 나눗셈 인자 / 이미지 픽셀의 표준편차
				# output size : 출력 이미지 사이즈
				# mean : mean substraction시 RGB 채널마다 각각 뺄셈을 적용할 R, G, B 값 정의 / BGR 채널로 구성된 이미지의 경우 R, B 채널을 순서를 바꾸도록 swapRB = True로 해야함
				# swapRB : BGR 채널 사용시 R, B 채널을 순서를 바꾸는 여부 결정
				# crop : 이미지 사이즈가 변한후 잘라낼지 여부를 결정

				detector.setInput(imageblob)
				# setInput : Neural Network의 입력값 형태를 결정함
				detections = detector.forward()		
				# forward : setInput에서 정의된 입력값을 Neural Network에 적용하고, 결과값을 받음
		 
				if len(detections) > 0:#detection에 하나 이상이 들어있을때 (사진에 얼굴이 하나 이상 있는 경우)

					k = np.argmax(detections[0, 0, :, 2]) # 감지 결과 중 가장 유사도가 높은값에 해당되는 index를 저장함 
					confidence = detections[0, 0, k, 2] # 감지 결과 중 가장 유사도가 높은값에 해당되는 index의 confidence값에 불러옴
					if confidence > compare_confidence_val:

						box = detections[0, 0, k, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")
						# 얼굴 감지결과를 그리는 박스가 프레임 밖으로 나가는 것을 방지한다.

						face = image[startY:endY, startX:endX]
						(fH, fW) = face.shape[:2]

						if fW < 20 or fH < 20:
							continue # 너비또는 폭이 20이하일때 (너무 멀리있는 얼굴에 대한 예외처리)
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
	# 'data' dictionary 변수에서 'names' key value에 해당되는 값 (학습 대상의 이름)을 라벨 변환하여 불러옴

	print('[INFO] Training model')
	recognizer = SVC(C=1.0, kernel='linear', probability=True)
	# SVM (Support Vector Machine)에 대해 정의함
	# C : 최적화 계수 
	# kernel : 커널 형태
	# probability : 분류 확률 연산 여부 / fit 함수 호출 전 정의해야함

	recognizer.fit(data['embeddings'], labels)
	# fit : 주어진 학습 데이터와 라벨을 기준으로 Fitting (데이터간 경계선을 형성)
	# 'data' dictionary 변수에서 학습 데이터에 대한 128d 값으로 저장된 'embedding' 값을 가지고 학습함

	f = open('face_detection/output/le.pickle', 'wb')
	f.write(pickle.dumps(le))
	f.close()

	f = open('face_detection/output/recognizer.pickle', 'wb')
	f.write(pickle.dumps(recognizer))
	f.close()

	protopath = 'face_detection/DNN_module/deploy.prototxt.txt'
	detector_modelpath = 'face_detection/DNN_module/res10_300x300_ssd_iter_140000.caffemodel'
	embedder_modelpath = 'face_detection/DNN_module/OpenFace.nn4.small2.v1.t7'

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
	target_img_path = 'face_detection/target_img/Son_Na_Eun_7.jpg'

	train_image_width = 600
	train_image_height = 0

	compare_confidence_val = 0.3

	image = cv2.imread(target_img_path)
	(h, w) = image.shape[:2]

	train_image_height = int((train_image_width * h)//w)
	image = cv2.resize(image, (train_image_width, train_image_height))
	(h, w) = image.shape[:2]

	imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

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

			for a in range(len(predictions)):
				pred_text = "{} : {:.2f}%".format(name[a], probability[a] * 100)
				print(pred_text)

			text = "{} : {:.2f}%".format(name, probability * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10

			cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
			cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)

	cv2.imshow('Output', image)
	cv2.waitKey(0)

