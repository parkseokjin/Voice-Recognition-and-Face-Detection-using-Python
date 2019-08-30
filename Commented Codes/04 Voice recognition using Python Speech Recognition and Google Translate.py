import speech_recognition as sr#음성인식을 도와주는 speech_recognition모듈을 가져오고,앞으로 sr이라 칭한다.
from os import path #os에서 파일 경로처리를 도와줄 path모듈을 가져온다.

AUDIO_FILE = path.join(path.dirname(path.realpath('/home/odroid/Workspace/Voice_Recognition_API/voice_1.py')),"voice_1.wav") #path.join:경로를 병합하여 새 경로를 생성하는 함수
							 #path.dirname:경로중 디렉토리명만 얻기
							 #path.realpath:file이 symbolic link일 경우 원본 위치를 찾아준다.
							 #symbolic link:디렉터리에 대한 참조를 포함하고 있는 파일,원본 대상으로부터 독립적으로 존재한다.
							 
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source: #AUDIO_FILE을 인식해 source라는 이름으로 저장
	audio = r.record(source) #source에 해당하는 모든 음성을 저장

try:
	print("Google Speech Recognition thinks you said" + r.recognize_google(audio, language="ko-KR")+ "in korean") #google speech recognition을 통해 audio를 입력받,language='ko-KR':language를 한국어로 해석한다.
except sr.UnknownValueError: #이해하지 못한 음성에 대하여
	print("Google Speech Recognition could not understand audio")
except sr.RequestError as e: #google speech recognition에 음성 해석에 대한 요청이 실패했을때. 
	print("Could nit request results from Google Speech Recognition service; {0}".format(e))



#try,except문:try블록 수행중 오류가 발생하면 except블록이 수행된다.하지만 try블록에서 오류가 발생하지 않으면 except블록은 수행되지 않는다.다음과 같이 except문이 여러개일때는 먼저 일어난 오류에 대해서만 오류를 출력한다.
