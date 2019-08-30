import numpy as np #numpy라는 모듈을 불러오고,np라고 부르기로 한다. 
import sys #sys라는 모듈을 불러온다.
sys.path.append('/usr/local/python/cv2/python-2.7') #모듈을 불러와 사용한다.
import cv2 #cv2라는 모듈을 불러온다.
print(sys.version)
print(cv2.__version__)

test_mode = 1 # 0 : ~~~ 모드 / 정적 이미지 처리
			  # 1 : ~~~ 모드 / 카메라 실시간 이미지 처리

if(test_mode == 0): #test_mode=0일때

    image = cv2.imread("image/test.jpg", cv2.IMREAD_ANYCOLOR) #image/test.jpg에 대한 이미지를 읽고, 그것을 image 변수에 대입한다.

    if(image is not None): # image값이 none이 아닐 때

        cv2.imshow("test_image_1", image) # 이미지를 보여준다
        cv2.waitKey(0) #어떤 키를 누를때 까지 기다린다
        cv2.destroyAllWindows()#모든 창을 닫는다.

    elif(image is None): #image값이 none일때

        print('Image Empty') #image empty출력

elif(test_mode == 1): #test_mode=1일때

    capture = cv2.VideoCapture(0) #0번째 캠에대한 영상을 capture변수에 대입한다.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #가로폭640 
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #세로폭480

    if(capture.isOpened()):#영상이 열렸을때
        print('video cam is available')
        while(capture.isOpened()):#영상이 열린동안
            ret, frame = capture.read() #비디오를 한 프레임씩 읽고, 제대로 읽으면 ret값에 true,실패하면 false가 들어간다.frame에는 읽은 프레임이 출력된다
            if(ret==True):#ret이 true일때
                cv2.imshow('video frame', frame)#video frame이라는 창에 frame을 출력한다
                if(cv2.waitKey(1) > 0):break#어떤 키를 누르기 전까지 대기한다.
        capture.release()#오픈한 캡쳐를 삭제
        cv2.destroyAllWindows()#모든 창을 닫는다

    else: #영상이 열리지 않았을때
        print('It has failed to open video cam') 


