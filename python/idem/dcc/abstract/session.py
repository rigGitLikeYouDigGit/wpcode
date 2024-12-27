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
import orjson


from wplib.network import libsocket

from idem import getIdemDir
from idem.dcc.abstract.command import *


def getPortDataPathMap()->dict[int, Path]:
	"""may not all be active"""
	return {int(i.name.split("_")[0]) : i
	                     for i in DCCIdemSession.sessionFileDir().iterdir()}

def getActivePortDataPathMap()->dict[int, Path]:
	activePortMap = libsocket.localPortSocketMap()
	result = {}
	for k, v in getPortDataPathMap().items():
		if not k in activePortMap:
			continue
		if activePortMap[k].type == "TCP" and activePortMap[k].state == "LISTENING":
			result[k] = v
	return result

def clearInactiveDataFiles():
	activeMap = getActivePortDataPathMap()
	for k, p in getPortDataPathMap().items():
		if not k in activeMap:
			p.unlink()


class SlotRequestHandler(socketserver.StreamRequestHandler):
	request : socket.socket
	def handle(self):
		"""call back to slots on server
		first deserialise from bytes with orjson"""
		log(self.server.socket.getsockname()[1], "handling request")
		#req = self.rfile.readline()
		# req = self.rfile.read()
		# # if req.startswith(libsocket.LEN_PREFIX_CHAR) or req.startswith(libsocket.LEN_PREFIX_CHAR.encode("utf-8")):
		# # 	req = req[4:]
		# req = req[4:] # strip the length prefix char from libsocket.sendMsg
		req = libsocket.recvMsg(self.request)
		#data : dict = orjson.loads(req)
		data = req

		# if the sender was this request's own server, don't respond to it
		if data["sender"][0] == self.server.socket.getsockname()[1]:
			log("not responding to own message") # works
			return
		if "t" in data:
			data = IdemCmd(data)
		log(self.server.socket.getsockname()[1], "handle data", data)

		for i in getattr(self.server, "slotFns", ()):
			try:
				i(self, data)
			except Exception as e:
				log("error handling req", data, "for server", self.server)
				traceback.print_exc()

	def sendResponse(self, msg:dict):
		"""send the given message dict back along the open socket"""
		log(self.server.socket.getsockname()[1], "send response", msg)
		libsocket.sendMsg(self.request, msg)

class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
	def server_bind(self):
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		super().server_bind()

class DataFileServer:
	"""base class for processes running a port server,
	with a data file in a set folder to declare themselves"""

	@classmethod
	def serverCls(cls)->type[socketserver.ThreadingTCPServer]:
		#return socketserver.ThreadingTCPServer
		return ReusableThreadingTCPServer


	def __init__(self, name="", linkTo=""):
		#self.uuid = str(uuid4())[:4]
		self.name = name

		self.loopThread : threading.Thread = None
		portId = libsocket.getFreeLocalPortIndex()
		# self.uuid = str(portId)
		self.server = self.serverCls()(
			("localhost", portId),
			SlotRequestHandler
		)
		self.uuid = str(self.server.socket.getsockname()[1])
		self.server.slotFns = [self.handleMessage]

		# create starting data files
		#self.clear() # get rid of any leftover files
		self.liveData = {
			"id" : self.uuid,
			"processId" : os.getpid(),
			"connected" : None # central bridge connected
		}
		self.writeSessionFileData(self.liveData)

		self.replicatedData = {
			"cameraTickTime" : 1.0/24.0
		} # consider just propagating this across all sessions -
		# for example, camera update speed

		if linkTo: # immediately link to a known session
			pass
		pass

	def onReplicatedDataChanged(self):
		"""but then do we pass in the keys that have changed, and the deltas,
		or do we have each be a discrete object that we hook signals to,
		and then we're back to doing reactive UI stuff except now it's not just
		data structures, it's coming over a network.
		I hate coding so much man
		"""
		pass

	def updateConnectedBridge(self, bridgeId):
		data = self.liveData
		data["connection"] = bridgeId
		self.writeSessionFileData(data)

	def availableBridgeSessions(self)->dict[int, str]:
		bridgeFiles = {k : v for k, v in self.portToFileMap().items() if v.name.split("_")[1].startswith("bridge")}
		activeMap = getActivePortDataPathMap()
		return {k : v for k, v in bridgeFiles.items() if k in activeMap}

	def availableDCCSessions(self)->dict[int, str]:
		bridgeFiles = {k : v for k, v in self.portToFileMap().items() if not v.name.split("_")[1].startswith("bridge")}
		activeMap = getActivePortDataPathMap()
		return {k : v for k, v in bridgeFiles.items() if k in activeMap}

	def connectedBridgeId(self):
		return self.sessionFileData().get("connection")

	def clear(self):
		"""clear up all data and pipe files for this session"""
		if self.loopThread:
			self.loopThread.join()
		#self.server.shutdown()
		if not self.sessionFileDir().is_dir():
			return
		for i in tuple(self.sessionFileDir().glob(self.uuid + "_*")):
			i.unlink(missing_ok=True)
		#if self.server
		#self.server.server_close()


	def __del__(self):
		self.clear()
		pass

	@classmethod
	def uuidToFileMap(cls)->dict[str, Path]:
		fileMap = {}
		for i in cls.sessionFileDir().iterdir():
			uid = i.name.split("_")[0]
			fileMap[uid] = i
		return fileMap

	@classmethod
	def portToFileMap(cls) -> dict[str, Path]:
		return {int(k) : v for k, v in cls.uuidToFileMap().items()}
	@classmethod
	def uuidToFileData(cls) -> dict[str, dict[str, Path]]:
		return {k : cls.sessionFileData(None, v)
		        for k, v in cls.uuidToFileMap().items()}

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
		return f"{self.uuid}_{self.name}"
	def sessionFilePath(self)->Path:
		return self.sessionFileDir() / (self._sessionFileName() + ".json")

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

	def handleMessage(self, handler:SlotRequestHandler, msg:dict):
		"""handle incoming messages from bridge"""
		raise NotImplementedError
		# if data["r"] : # wants a response
		# 	response = self.message({"my" : "response"}, wantResponse=False)
		# 	libsocket.sendMsg(handler.request, response)

	def portId(self)->int:
		return int(self.uuid)

	def message(self, data:dict, wantResponse=False)->MsgDict:
		"""add message attributes to dict"""
		data["sender"] = (self.portId(), self.name)
		data["r"] = wantResponse
		return data

	def send(self, msg:dict, toPort=None):
		"""send a message from this session back to the bridge

		TODO: unsure if there's a point in sending a direct response back
			this way, maybe for confirmation of publish, in case there's an
			error in the export somehow?
			but naively I'd rather catch and filter that as a general incoming message
		"""
		toPort = toPort if toPort is not None else self.portId()
		log(self._sessionFileName(), "send to", toPort, msg)
		with socket.socket() as sock:
			if msg["r"]:  # allow longer for response
				sock.settimeout(1.0)
			else:
				sock.settimeout(1.0)

			try:
				sock.connect(("localhost", toPort))
			except ConnectionRefusedError:
				log(self.uuid, f"IDEM ERROR: no listening port on {toPort}, aborting send")
				return
			except socket.timeout:
				# later consider retrying any 'important' commands?
				log(self.uuid, f"IDEM ERROR: port {toPort} timed out, aborting send")
				return
			libsocket.sendMsg(sock, msg)

			if msg["r"]: # wait for response
				log(self._sessionFileName(), "wait for reply")
				result = libsocket.recvMsg(sock)
				log(self._sessionFileName(), "got reply", result)
				return result

	# def sendCmd(self, msg:DCCCmd, toPort=None):
	# 	"""send the given command across the network to synchronise idem data"""

	def getSessionIdData(self)->SessionIdData:
		return SessionIdData(id=(self.portId(), self.name),
		                     dcc=self.dccType)

class SessionIdData(T.TypedDict):
	id : tuple[int, str]
	dcc : str



class DCCIdemSession(DataFileServer):
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
		return self.name

	def sessionNiceName(self):
		s = f"{self.uuid}_{self.dccType}"
		try:
			if self._getDCCFileName():
				s += ("_" + self._getDCCFileName())
		except NotImplementedError:
			pass
		return s


	def handleMessage(self, handler:SlotRequestHandler, msg:dict):
		"""handle incoming messages from bridge"""
		log(self._sessionFileName(), "handle", msg)
		if isinstance(msg, ReplicateDataCmd):
			self.replicatedData = msg["data"]
			return
		if isinstance(msg, ConnectToBridgeCmd):
			self.updateConnectedBridge(msg["sender"])
			handler.sendResponse(
				self.message({"connectedSession" : self.getSessionIdData()}, # should we have a function to make this response?
				             wantResponse=False)
			)
			return
		#return super().handleMessage(handler, data)



	def getSessionCamera(self):
		"""return a DCC camera to link to idem,
		updating views simultaneously across programs.
		If this works it'll be so cool"""




@dataclass
class ChildSessionData:
	port : int
	server : socketserver.ThreadingTCPServer
	serverThread: threading.Thread
	dccTypeName : str = ""



class IdemBridgeSession(DataFileServer):
	"""central coordinating system, sitting in separate thread/process,
	for now just echoing events back to all open sessions

	write all attached processes into datafile, so other bridges
	don't try and steal them

	- manage named blocks of shared memory

	"""
	dccType = "bridge"

	def __init__(self, name="bridge"):
		super().__init__(name)
		self.linkedSessions : dict[int, ChildSessionData] = {}

		self.server.slotFns = [
			lambda *args, **kwargs : self.handleMessage(
				*args, fromPort=self.portId(), **kwargs)
		                  ]

		self.liveData = {
			"id" : self.uuid,
			"processId" : os.getpid(),
			"connected" : {} # type:dict[int, SessionIdData]
		}
		self.writeSessionFileData(self.liveData)

	def connectToSocket(self, portId:int):
		"""create a new server listening on the given child socket"""
		server = self.serverCls()(
			("localhost", portId),
			SlotRequestHandler
		)
		server.slotFns = [
			lambda *args, **kwargs : self.handleMessage(
				*args, fromPort=portId, **kwargs)
		                  ]
		thread = threading.Thread(target=server.serve_forever)
		self.linkedSessions[portId] = ChildSessionData(
			portId, server, serverThread=thread)

		# ask server for some other information
		result = self.send(self.message(
			ConnectToBridgeCmd(), wantResponse=True),
			toPort=portId
		)
		if not result:
			raise RuntimeError("could not connect to child session ", portId)
		if "connectedSession" in result:
			log("successfully connected child session", result["connectedSession"])
			# json keys apparently can't be integers, only strings
			# been working with this format 7 years, only learned that today
			self.liveData["connected"][str(portId)] = result["connectedSession"]
			self.writeSessionFileData(self.liveData)

	def disconnectSocket(self, portId:int):
		"""remove connection to the given port -
		shutdown listening server
		halt listening thread
		remove from connections"""
		self.linkedSessions[portId].server.shutdown()
		self.linkedSessions.pop(portId)
		self.liveData["connected"].pop(str(portId))
		self.writeSessionFileData(self.liveData)

	def send(self, msg:dict, toPort=None):
		if toPort is None:
			for k, server in self.linkedSessions.items():
				# can't get response from general message
				msg["r"] = False
				self.send(msg, toPort=k)
			return
		return super().send(msg, toPort)

	def handleMessage(self, handler:SlotRequestHandler, msg:(IdemCmd, dict), *args,
	                      fromPort=0, **kwargs):
		"""if fromPort is None, probably came from
		this bridge session"""
		if isinstance(msg, ReplicateDataCmd):
			self.replicatedData = msg["data"]
			return
		if isinstance(msg, ConnectToSessionCmd):
			self.connectToSocket(msg["targetPort"])
			handler.sendResponse({"connectedSession" : self.getSessionIdData()})
			return
		if isinstance(msg, DisconnectSessionCmd):
			self.disconnectSocket(msg["targetPort"])
			return
