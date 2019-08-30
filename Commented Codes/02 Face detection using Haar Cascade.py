import numpy as np #numpy라는 모듈을 불러오고, np라고 부르기로 한다.
import sys #sys라는 모듈을 불러온다.
sys.path.append('/usr/local/python/cv2/python-2.7') # Library Path 경로 읽는 순서를 변경하기 위해 Path 변수에 대한 재정의 / 모듈을 불러와 사용한다.
import cv2 #cv2라는 모듈을 불러온다.

face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
# Cascade 알고리즘 기반 이미 학습된 얼굴 인식 모델 객체를 형성한다.

eye_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml')
# Cascade 알고리즘 기반 이미 학습된 얼굴 인식 모델 객체를 형성한다.

test_mode = 1 # 0 : 정적 이미지에 대한 Haar Cascade 얼굴 인식
			  # 1 : 카메라 실시간 이미지에 대한 Haar Cascade 얼굴 인식

if(test_mode == 0):
	img = cv2.imread('image/img.jpg') #상대경로를 통해 얻은 img.jpg파일을 읽고 img변수에 넣는다.
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #img값에 대한 컬러 이미지를 GrayScale 이미지로 바꿔서 넣어준다.
        
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
	# 입력된 이미지에 대해 얼굴을 감지한다.
	# 여기서 1.3은 scalefactor인데,30%만큼 사진의 사이즈를 감소시키는 것이다.
	# 5는 minneighbors인데, 눈이 감지되었을때 얼마나 오랫동안 있어야하는지 알려주는 것이다. 값이 높을수록 정확도는 떨어지고,안정성이 올라간다.

	if(img is None):
		print('Empty img')
	for (x,y,w,h) in faces: # 검출된 얼굴 이미지를 튜플로 리턴한다.x,y는 얼굴의 좌상단.w,h는 가로와 세로길이이다.
		img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#얼굴 이미지를 청색 사각형으로 그려넣는다.
		roi_gray = gray[y:y+h, x:x+w] #얼굴만 잘라낸 grayscale이미지
		roi_color = img[y:y+h, x:x+w] #얼굴만 잘라낸 컬러 이미지
		eyes = eye_cascade.detectMultiScale(roi_gray) #얼굴만 잘라낸 grayscale이미지에 대해서 눈을 인식한다.
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #얼굴 인식과 같은 방식으로 사각형을 생성한다. 단, 사각형은 초록색이다(0.255.0)

	cv2.imshow('img',img) #얼굴인식 및 눈 인식 결과를 보여준다
	cv2.waitKey(0)#키 입력있을 때까지 대기
	cv2.destroyAllWindows()#모든 창을 닫는다.

elif(test_mode == 1):
	
	capture = cv2.VideoCapture(0) #0번째 카메라에 대한 영상 스트리밍 정보 접근 / 카메라 디바이스 드라이버를 점유함
	capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)#가로폭 640
	capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)#세로폭480	
	if(capture.isOpened()): # 카메라 디바이스 드라이버가 성공적으로 활성화되었을때
		print('video cam is available')
		while(capture.isOpened()): #카메라가 활성화된 동안 지속적으로 카메라 프레임을 실시간으로 읽고 처리하여, 결과 이미지를 보여줌
			ret, frame = capture.read() #카메라 영상을 한 프레임씩 읽고, 제대로 읽으면 ret값에 true,아니면 false대입, frame에는 읽은 이미지 프레임이 담겨있다.

			if(ret==True):
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #읽은 프레임이 담겨인는 frame변수에 대한 컬러 이미지를 grayscale이미지로 바꿔서 gray에 담아준다.
				faces = face_cascade.detectMultiScale(gray, 1.3, 5)
				# 입력된 이미지에 대해 얼굴을 감지한다.
				# 여기서 1.3은 scalefactor인데,30%만큼 사진의 사이즈를 감소시키는 것이다.
				# 5는 minneighbors인데, 눈이 감지되었을때 얼마나 오랫동안 있어야하는지 알려주는 것이다. 값이 높을수록 정확도는 떨어지고,안정성이 올라간다.

				print(faces)#얼굴의 위치를 리스트 값으로 출력해준다.

				for (x,y,w,h) in faces: # 검출된 얼굴 이미지를 튜플로 리턴한다.x,y는 얼굴의 좌상단.w,h는 가로와 세로길이이다.
					frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #얼굴 이미지를 청색 사각형으로 그려넣는다.
					roi_gray = gray[y:y+h, x:x+w] #얼굴만 잘라낸 grayscale이미지
					roi_color = frame[y:y+h, x:x+w] #얼굴만 잘라낸 컬러 이미지
					
				cv2.imshow('image', frame)
        
			if(cv2.waitKey(1) > 0): break

		capture.release() #점유하고 있는 카메라 대해 점유를 해제함
		cv2.destroyAllWindows()

	else:
		print('It has failed to open video cam') 
