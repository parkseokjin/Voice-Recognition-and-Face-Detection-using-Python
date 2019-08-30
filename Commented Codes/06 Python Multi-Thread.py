from concurrent.futures import ThreadPoolExecutor#multithread을 이용한 병렬처리를 도와줄 모듈을 가지고 온다.

def main1():#main1함수를 정의한다.
	count = 1
	while count != 10:
		print("main1")
		count = count + 1 #count를 1부터 시작시켜 1씩 증가시킬때, count가 10이 되기 전까지 'main1'문장을 출력.
	return 'hello world'#'hello world'문장 출력

def main2():#main2함수를 정의한다.
	
	count = 1
	while count != 10:
		print("main2")
		count = count + 1#count를 1부터 시작시켜 1씩 증가시킬때, count가 10이 되기 전까지 'main2'문장을 출력.
	return 'HELLO WORLD'#'HELLO WORLD'출렦

#---------------------------------------------#

pool_executor = ThreadPoolExecutor(1)#ThreadPoolExecutor객체를 생성한다.

pool = pool_executor.submit(main1)#submit()을 통해 병렬 처리 실행을 원하는 함수를 입력해 주면 해당함수에 대해 별도의 프로세스를 할당함.
#각각의 함수에 대해 독립적인 프로세스를 할당함으로서 병렬적으로 동시 처리가 구현됨.

if pool.done():#done():처리되고 있는 함수가 종료되었는지 처리되었는지 확인 가능
	print("pool result : " + pool.result())
else:
	print('not yet')

#---------------------------------------------#

pool2_executor = ThreadPoolExecutor(2)

pool2 = pool2_executor.submit(main2)

if pool2.done():
	print("pool2 result : " + pool2.result())
else:
	print('NOT YET')
#이 코드를 실행할 때 처음에 not yet이 뜨는 이유는 main1과 main2함수가 아직 실행중일 때 detech되어서 pool.done()을 만족하지 못했기 때문이다. 






