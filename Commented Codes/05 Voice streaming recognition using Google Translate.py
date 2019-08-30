import speech_recognition as sr
from os import path
import pyaudio
from six.moves import queue
import time

RATE = 16000 #녹음용 값의 rate값은 16kHz.
CHUNK=int(RATE/10)#버퍼는 1600(녹음시 지연시간)

class MicrophoneStream(object):
	def __init__(self, rate, chunk):
       	    self._rate = rate
       	    self._chunk = chunk
	    self._buff = queue.Queue()
       	    self.closed = True
	def __enter__(self):
	    self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,#16비트,모노로 마이크 열기
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer, #fill_buffer에 버퍼가 쌓이면 이곳이 호출된다.
        )        
       	    self.closed = False
            return self
	def __exit__(self,type,value,traceback):#클래시 종료시
											#pyaudio종료
	    self._audio_stream.stop_stream()
            self._audio_stream.close()
	
	    self.closed = True
	    self._buff.put(None)
            self._audio_interface.terminate()
	
	def _fill_buffer(self, in_data, frame_count, time_info, status_flags):#마이크 버퍼가 쌓이면 
            self._buff.put(in_data)#마이크 입력 받으면 큐에 넣고 리턴.
            return None, pyaudio.paContinue

	def generator(self):#클래스 종료될때까지 무한 루프 돌리기
		while not self.closed:
			chunk = self._buff.get()#데이터를 기다린다.
			if chunk is None:#데이터가 없다면 진행할 수 없음.
				return
			data = [chunk]#data에 마이크 입력을 받는다.
			while True:
				try:
					chunk = self._buff.get(block=False)#데이터가 더 있는지 확인
					if chunk is None:
						return
					data.append(chunk)#데이터를 추가한다.
				except queue.Empty:
					break
			yield b''.join(data) #마이크 데이터 리턴.
	def main():
		with MicrophoneStream(RATE,CHUNK) as stream:
			audio_generator =  stream.generator()
			for i in range(1000):#마이크 데이터 1000번 가져오고 나간다.
				for x in audio_generator:
					print(x)
				time.sleep(0.001)
			AUDIO_FILE = data
			r = sr.Recognizer()
			with sr.AudioFile(AUDIO_FILE) as source:
				audio = r.record(source)
			try:
				print("Google Speech Recognition thinks you said" + r.recognize_google(audio, language="ko-KR")+ "in korean")
			except sr.UnknownValueError: 
				print("Google Speech Recognition could not understand audio")
			except sr.RequestError as e:
				print("Could nit request results from Google Speech Recognition service; {0}".format(e))

	if __name__ == '__main__':
			main()

