from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
import time, os, threading, sys
from pathlib import Path
import win32pipe, win32file, pywintypes

"""
TODO: make this work across windows and linux at some point

abandoning named pipes entirely, I found the script below hanging at https://timgolden.me.uk/pywin32-docs/win32pipe__ConnectNamedPipe_meth.html
and I have no idea why, and no idea how to go about learning why.

sockets it is.

"""


# example functions by ChrisWue on SO
def pipe_server():
	print("pipe server")
	count = 0
	pipe = win32pipe.CreateNamedPipe(
		r'\\.\pipe\Foo',
		win32pipe.PIPE_ACCESS_DUPLEX,
		win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
		1, 65536, 65536,
		0,
		None)
	try:
		print("waiting for client")
		win32pipe.ConnectNamedPipe(pipe, None)
		print("got client")

		while count < 10:
			print(f"writing message {count}")
			# convert to bytes
			some_data = str.encode(f"{count}")
			win32file.WriteFile(pipe, some_data)
			time.sleep(1)
			count += 1

		print("finished now")
	finally:
		win32file.CloseHandle(pipe)


def pipe_client():
	print("pipe client")
	quit = False

	while not quit:
		try:
			handle = win32file.CreateFile(
				r'\\.\pipe\Foo',
				win32file.GENERIC_READ | win32file.GENERIC_WRITE,
				0,
				None,
				win32file.OPEN_EXISTING,
				0,
				None
			)
			res = win32pipe.SetNamedPipeHandleState(handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
			if res == 0:
				print(f"SetNamedPipeHandleState return code: {res}")
			while True:
				resp = win32file.ReadFile(handle, 64*1024)
				print(f"message: {resp}")
		except pywintypes.error as e:
			if e.args[0] == 2:
				print("no pipe, trying again in a sec")
				time.sleep(1)
			elif e.args[0] == 109:
				print("broken pipe, bye bye")
				quit = True

#
# if __name__ == '__main__':
# 	if len(sys.argv) < 2:
# 		print("need s or c as argument")
# 	elif sys.argv[1] == "s":
# 		pipe_server()
# 	elif sys.argv[1] == "c":
# 		pipe_client()
# 	else:
# 		print(f"no can do: {sys.argv[1]}")

class WindowsNamedPipe:
	"""one end of the pipe, or at least
	one entity capable of modifying it
	"""
	def __init__(self, path:Path,
				 readLineFns:list[T.Callable[[str], None]],
				 delay=1.0
				 ):
		self.path = path
		self.readLineFns = readLineFns
		self.delay = delay
		self.listenThread : threading.Thread = None
		self.quit = False


	def listen(self):
		self.quit = False

		while not self.quit:
			try:
				handle = win32file.CreateFile(
					# r'\\.\pipe\Foo',
					self.path,
					win32file.GENERIC_READ | win32file.GENERIC_WRITE,
					0,
					None,
					win32file.OPEN_EXISTING,
					0,
					None
				)
				res = win32pipe.SetNamedPipeHandleState(handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
				if res == 0:
					#print(f"SetNamedPipeHandleState return code: {res}")
					time.sleep(self.delay)
				while True:
					resp = win32file.ReadFile(handle, 64 * 1024)
					#print(f"message: {resp}")
					for fn in self.readLineFns:
						fn(resp)
			except pywintypes.error as e:
				if e.args[0] == 2:
					print("no pipe, trying again in a sec")
					time.sleep(self.delay)
				elif e.args[0] == 109:
					print("broken pipe, bye bye")
					self.quit = True

	def listenThreaded(self):
		self.listenThread = threading.Thread(target=self.listen)
		self.listenThread.start()
		return self.listenThread

	def __del__(self):
		self.quit = True
		if self.listenThread:
			self.listenThread.join()


	def write(self, data:str):
		count = 0
		log("before create")
		pipe = win32pipe.CreateNamedPipe(
			#r'\\.\pipe\Foo',
			self.path,
			#win32pipe.PIPE_ACCESS_DUPLEX,
			win32pipe.PIPE_ACCESS_OUTBOUND,
			win32pipe.PIPE_TYPE_MESSAGE, #| win32pipe.PIPE_READMODE_MESSAGE, #| win32pipe.PIPE_WAIT,
			1, 65536, 65536,
			0,
			None)
		log("before connect")
		win32pipe.ConnectNamedPipe(pipe, None)
		log("before write")
		win32file.WriteFile(pipe, data.encode())
		log("before close")
		win32file.CloseHandle(pipe)

if __name__ == '__main__':

	def readLineFn(*args, **kwargs):
		print("read line", args, kwargs)

	path = Path("F:/wp/code/python/idem/localsessions/1595_python_")
	path = r'\\.\pipe\Foo'

	log("before read")
	listener = WindowsNamedPipe(
		path,
		readLineFns=[readLineFn]
	)

	# listener.listen()
	log("before listen")
	listener.listenThreaded()

	log("before init")
	writer = WindowsNamedPipe(
		path,
		readLineFns=[readLineFn]
	)
	log("before write")
	writer.write("test_data")

	writer.write("second test data")





