import speech_recognition as sr
from os import path 
AUDIO_FILE = path.join(path.dirname(path.realpath('/home/odroid/Workspace/Voice_Recognition_API/voice_2.py')),"voice_2.wav") 
							 
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source: 
	audio = r.record(source) 
try:
	print("Google Speech Recognition thinks you said" + r.recognize_google(audio, language="ko-KR")+ "in korean") 
except sr.UnknownValueError: 
	print("Google Speech Recognition could not understand audio")
except sr.RequestError as e: 
	print("Could nit request results from Google Speech Recognition service; {0}".format(e))




