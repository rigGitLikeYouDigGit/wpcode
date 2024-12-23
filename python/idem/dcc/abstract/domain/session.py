from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import os, sys
from collections import defaultdict
import threading, multiprocessing, socket, socketserver, traceback
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass

from idem import getIdemDir

import orjson

import win32pipe

from wplib.network import libsocket

class SlotRequestHandler(socketserver.StreamRequestHandler):
	request : socket.socket
	def handle(self):
		"""call back to slots on server
		first deserialise from bytes with orjson"""
		#log("handling request")
		#req = self.rfile.readline()
		req = self.rfile.read()
		# if req.startswith(libsocket.LEN_PREFIX_CHAR) or req.startswith(libsocket.LEN_PREFIX_CHAR.encode("utf-8")):
		# 	req = req[4:]
		req = req[4:] # strip the length prefix char from libsocket.sendMsg
		data = orjson.loads(req)
		for i in getattr(self.server, "slotFns", ()):
			try:
				i(data)
			except Exception as e:
				log("error handling req", data, "for server", self.server)
				traceback.print_exc()


class DCCIdemSession:
	"""domain-side class for access within a dcc,
	handles linking to a bridge idem process,
	declares itself as open if no bridge is found

	session is persistent with whole lifetime of DCC, or
	until this object is cleared - survive through new files, renaming, etc

	files in localsessions are named "akda_maya_myMayaFile" -
		first 4 letters are uuid, and those are all we need to identify,
		the rest are just for ease of inspection
	"""

	dccType = "python" # override with "maya", "houdini", etc

	def __init__(self, sessionTempName="", linkTo=""):
		#self.uuid = str(uuid4())[:4]
		self.sessionTempName = sessionTempName

		self.loopThread : threading.Thread = None
		portId = libsocket.getFreeLocalPortIndex()
		# self.uuid = str(portId)
		self.server = socketserver.ThreadingTCPServer(
		#self.server = socketserver.TCPServer(
			("localhost", portId),
			#("", 0),
			SlotRequestHandler
			#socketserver.BaseRequestHandler
		)
		self.uuid = str(self.server.socket.getsockname()[1])

		self.server.slotFns = [self.handleMessage]

		# create starting data files
		#self.clear() # get rid of any leftover files
		self.writeSessionFileData(self.liveData())
		# os.mkfifo(self.inPipeFilePath())
		# os.mkfifo(self.outPipeFilePath())


		if linkTo: # immediately link to a known session
			pass
		pass

	def liveData(self)->dict:
		return {
			"id" : self.uuid,
			"processId" : os.getpid()
		}

	def clear(self):
		"""clear up all data and pipe files for this session"""
		if self.loopThread:
			self.loopThread.join()
		if not self.sessionFileDir().is_dir():
			return
		for i in tuple(self.sessionFileDir().glob(self.uuid + "_*")):
			i.unlink(missing_ok=True)
		#if self.server
		#self.server.server_close()
		self.server.shutdown()


	def __del__(self):
		self.clear()
		pass

	@classmethod
	def uuidToFileMap(cls)->dict[str, dict[str, Path]]:
		fileMap = defaultdict(dict)
		for i in cls.sessionFileDir().iterdir():
			uid = i.name.split("_")[0]
			if i.name.endswith("_IN"):
				fileMap[uid]["in"] = i
			elif i.name.endswith("_OUT"):
				fileMap[uid]["out"] = i
			else:
				fileMap[uid]["data"] = i
		return fileMap

	@classmethod
	def uuidToFileData(cls) -> dict[str, dict[str, Path]]:
		fileMap = defaultdict(dict)
		for i in cls.sessionFileDir().iterdir():
			uid = i.name.split("_")[0]
			if i.name.endswith("_IN"):
				fileMap[uid]["in"] = i
			elif i.name.endswith("_OUT"):
				fileMap[uid]["out"] = i
			else:
				fileMap[uid]["data"] = cls.sessionFileData(None, i)
		return fileMap

	def sessionFileData(self, path:Path=None)->dict:
		path = path or self.sessionFilePath()
		if not path.is_file():
			return None
		with open( path, "rb") as f:
			return orjson.loads(f.read())

	def writeSessionFileData(self, data:dict, path:Path=None):
		path = path or self.sessionFilePath()
		log("write session file data", path, data)
		with open(path, "wb") as f:
			f.write(orjson.dumps(data))

	@classmethod
	def sessionFileDir(cls)->Path:
		return getIdemDir() / "localsessions"

	def _sessionFileName(self)->str:
		return f"{self.uuid}_{self.dccType}_{self._getDCCFileName()}"
	def sessionFilePath(self)->Path:
		return self.sessionFileDir() / (self._sessionFileName() + ".json")
	def _getDCCFileName(self)->str:
		"""current file name worked on by this DCC session
		or None if no file name set
		OVERRIDE THIS
		"""
		#raise NotImplementedError
		return self.sessionTempName

	def sessionNiceName(self):
		s = f"{self.uuid}_{self.dccType}"
		try:
			if self._getDCCFileName():
				s += ("_" + self._getDCCFileName())
		except NotImplementedError:
			pass
		return s


	def runThreaded(self):
		"""execute the below run() method in a child thread
		"""
		self.loopThread = threading.Thread(target=self.run, daemon=True)
		self.loopThread.start()

	def run(self):
		"""startup method that begins event loop, handles errors
		for now we just hack an event loop here
		"""
		self.server.serve_forever(poll_interval=1.0)

	def handleMessage(self, data):
		print(self.sessionNiceName() + "handle:", data)

	def portId(self)->int:
		return int(self.uuid)

	def send(self, data, waitForResponse=False):
		"""send a message from this session back to the bridge

		TODO: unsure if there's a point in sending a direct response back
			this way, maybe for confirmation of publish, in case there's an
			error in the export somehow?
			but naively I'd rather catch and filter that as a general incoming message
		"""
		with socket.socket() as sock:
			sock.connect(("localhost", self.portId()))
			#sock.sendall(bytes(data, 'utf-8'))
			libsocket.sendMsg(sock, orjson.dumps(data, ))

			if waitForResponse:
				return libsocket.recvMsg(sock)


def getPortDataPathMap()->dict[int, Path]:
	"""may not all be active"""
	return {int(i.name.split("_")[0]) : i
	                     for i in DCCIdemSession.sessionFileDir().iterdir()}

def getActivePortDataPathMap()->dict[int, Path]:
	return {k : v for k, v in getPortDataPathMap().items()
	        if libsocket.portIsOpen("localhost", k)}


@dataclass
class ChildSessionData:
	port : int
	server : socketserver.ThreadingTCPServer
	dccTypeName : str = ""


class IdemBridgeSession:
	"""central coordinating system, sitting in separate thread/process,
	for now just echoing events back to all open sessions

	write all attached processes into datafile, so other bridges
	don't try and steal them

	- manage named blocks of shared memory

	"""

	def __init__(self):
		self.linkedSessions : dict[int, ChildSessionData] = {}

	def connectToSocket(self, portId:int):
		server = socketserver.ThreadingTCPServer(
			("localhost", portId),
			SlotRequestHandler
		)
		server.slotFns = [
			lambda *args, **kwargs : self.handleMessageFrom(
				portId, *args, fromPort=portId, **kwargs)
		                  ]
		self.linkedSessions[portId] = ChildSessionData(
			portId, server)
		# ask server for some other information


	def sendMsg(self, data, toPort=None, waitForResponse=False):
		if toPort is None:
			for k, server in self.linkedSessions.items():
				self.sendMsg(data, toPort=k)
			return
		with socket.socket() as sock:
			sock.connect(("localhost", toPort))
			#sock.sendall(bytes(data, 'utf-8'))
			libsocket.sendMsg(sock, orjson.dumps(data, ))
			if waitForResponse:
				libsocket.recvMsg(sock)

	def handleMessageFrom(self, data, *args, fromPort:int=None, **kwargs):
		"""if fromPort is None, probably came from
		this bridge session"""



