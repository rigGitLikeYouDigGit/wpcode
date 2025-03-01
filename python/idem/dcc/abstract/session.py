from __future__ import annotations

import os, datetime, sys
import pprint
import threading, socket, socketserver, traceback, atexit
import time
from pathlib import Path
import orjson

from wplib.exchook import MANAGER
from wplib.network import libsocket

from idem import getIdemDir
from idem.dcc.abstract.command import *

MANAGER.activate()

def getPortDataPathMap()->dict[int, Path]:
	"""may not all be active"""
	return {int(i.name.split("_")[0]) : i
	                     for i in DCCIdemSession.sessionFileDir().iterdir()}

PING_CALL = "IDEMPING"
PING_RESPONSE = "IDEMPONG"

def getActivePortDataPathMap()->dict[int, Path]:
	"""this is not robust enough since some sockets hang around for unknown
	lengths of time -
	instead relying solely on the files in the localsessions folder"""
	# activePortMap = libsocket.localPortSocketMap()
	result = {}
	for k, v in getPortDataPathMap().items():
		sock = socket.socket()
		sock.settimeout(0.1)
		with sock:
			try:
				sock.connect(("localhost", k))
			except ConnectionRefusedError:
				continue
			except socket.timeout:
				continue
			libsocket.sendMsg(sock, PING_CALL)
			try:
				response = libsocket.recvMsg(sock)
				if response != PING_RESPONSE:
					continue
			except socket.timeout :
				continue

		result[k] = v
	return result
	return getPortDataPathMap()

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
		data = libsocket.recvMsg(self.request)
		if data == PING_CALL : # basic lookup signal to find out which sockets are active
			self.sendResponse(PING_RESPONSE)
			return
		#log(self.server.socket.getsockname()[1], "handling request")

		# if the sender was this request's own server, don't respond to it
		if data["s"][0] == self.server.socket.getsockname()[1]:
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
		if str(msg) != PING_RESPONSE:
			log(self.server.socket.getsockname()[1], "send response", msg)
		libsocket.sendMsg(self.request, msg)

class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
	daemon_threads = True
	def server_bind(self):
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		super().server_bind()

class DataFileServer:
	"""base class for processes running a port server,
	with a data file in a set folder to declare themselves

	we're really abusing threads here
	"""

	@classmethod
	def serverCls(cls)->type[socketserver.ThreadingTCPServer]:
		#return socketserver.ThreadingTCPServer
		return ReusableThreadingTCPServer

	# class-level module for the current session, in the current interpreter
	_session :DataFileServer = None
	@classmethod
	def session(cls)->(None, DataFileServer):
		return cls._session
	@classmethod
	def getSession(cls, name="abstractIdem")->DataFileServer:
		if not cls._session:
			cls._session = cls.bootstrap(name)
		return cls._session

	@classmethod
	def bootstrap(cls, sessionName="IDEM", start=True)->cls:
		"""load up an abstract IDEM session from standing start, set up
		ports, hook up idem camera, sets etc
		"""
		if cls.session():
			cls.session().clear()
		log("bootstrap", cls, "listening sockets:")
		# pprint.pp(libsocket.localPortSocketMap(), sort_dicts=True)
		# pprint.pp(getActivePortDataPathMap())
		clearInactiveDataFiles()
		newSession = cls(name=sessionName)
		if start:
			newSession.runThreaded()
		newSession.log("bootstrapped", sessionName, start)
		cls._session = newSession

		return newSession


	def __init__(self, name="", linkTo=""):
		#self.uuid = str(uuid4())[:4]
		self.name = name

		self.loopThread : threading.Thread = None
		self.heartbeatThread : threading.Thread = None
		self.heartbeatT = 0.5 # seconds
		self.shouldStopHeartbeat = False
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

		# ensure no matter what we always remove the file record for this session
		atexit.register(self.clear)
		#MANAGER.weakFns.add(self.onExceptHook)
		MANAGER.fns.append(self.onExceptHook)

	def onExceptHook(self, excType, val, tb):
		if excType in (KeyboardInterrupt, ):
			self.clear()

	def onHeartbeatTimeout(self, timeoutPorts:list[int]):
		self.log("connection ", timeoutPorts, "did not respond to heartbeat in time")

	def getHeartbeatPorts(self)->list[int]:
		"""get list of connected items to check with heartbeat"""
		raise NotImplementedError

	def heartbeatLoop(self):
		"""heartbeat timeout should mean the other end of the process is
		ended, without question - unlink this end of it
		"""
		while not self.shouldStopHeartbeat:
			# ping each one
			timeoutPorts = []
			for i in self.getHeartbeatPorts():
				sock = socket.socket()
				sock.settimeout(self.heartbeatT)
				with sock:
					try:
						sock.connect(("localhost", i))
					except (ConnectionRefusedError, socket.timeout):
						self.log("heartbeat could not connect to ", i)
						timeoutPorts.append(i)
						continue
					libsocket.sendMsg(sock, PING_CALL)
					try:
						response = libsocket.recvMsg(sock)
					except socket.timeout:
						self.log("heartbeat timedout waiting for response from", i)
						timeoutPorts.append(i)
						continue
					if response != PING_RESPONSE:
						self.log("heartbeat got wrong ping response message from", i)
						timeoutPorts.append(i)

			if timeoutPorts:
				self.onHeartbeatTimeout(timeoutPorts)
			time.sleep(self.heartbeatT)



	def log(self, *args):
		return log(datetime.datetime.now().time(), ": ", self._sessionFileName(), ": ", *args, framesUp=1)


	def onReplicatedDataChanged(self):
		"""but then do we pass in the keys that have changed, and the deltas,
		or do we have each be a discrete object that we hook signals to,
		and then we're back to doing reactive UI stuff except now it's not just
		data structures, it's coming over a network.
		I hate coding so much man
		"""
		pass

	def updateConnectedBridge(self, bridgeId, sendCmd=False):
		"""called on CHILD to drive BRIDGE"""
		if self.connectedBridgeId():
			raise RuntimeError("Disconnect session bridge before connecting to a new one")
		if sendCmd:
			log(self.portId(), "->", bridgeId, ": connect from session to bridge")
			result = self.send(self.message(
				ConnectToSessionCmd(targetPort=bridgeId),
				wantResponse=True
			),
				toPort=bridgeId
			)
			assert result, "Got no result from bridge after sending connectToSession CMD"
		data = self.liveData
		data["connected"] = bridgeId
		self.writeSessionFileData(data)

	def disconnectBridge(self, sendCmd=True):
		if not self.connectedBridgeId():
			return
		if sendCmd:
			self.send(
				self.message(DisconnectSessionCmd(targetPort=self.portId())),
				toPort=self.connectedBridgeId()
			)
		self.liveData["connected"] = None
		self.writeSessionFileData(self.liveData)

	@classmethod
	def availableBridgeSessions(cls)->dict[int, Path]:
		"""does not check if connected"""
		bridgeFiles = {k : v for k, v in cls.portToFileMap().items() if v.name.split("_")[1].startswith("bridge")}
		activeMap = getActivePortDataPathMap()
		return {k : v for k, v in bridgeFiles.items() if k in activeMap}

	@classmethod
	def availableDCCSessions(cls)->dict[int, Path]:
		"""does not check if connected
		rename to 'currentDCCSessions' or smt"""
		bridgeFiles = {k : v for k, v in cls.portToFileMap().items() if not v.name.split("_")[1].startswith("bridge")}
		activeMap = getActivePortDataPathMap()
		return {k : v for k, v in bridgeFiles.items() if k in activeMap}

	def connectedBridgeId(self):
		#return self.sessionFileData().get("connected")
		return self.liveData.get("connected")

	def clear(self):
		"""clear up all data and pipe files for this session"""
		self.log("CLEAR start", self.loopThread, self.server)
		self.shouldStopHeartbeat = True

		if self.sessionFileData() is not None:
			if self.connectedBridgeId():
				self.log("disconnecting bridge")
				self.disconnectBridge(sendCmd=True)

		if self.sessionFileDir().is_dir():
			for i in tuple(self.sessionFileDir().glob(self.uuid + "_*")):
				i.unlink(missing_ok=True)

		self._session = None
		if self.server:
			self.log("try server shutdown")
			#self.server.server_close()
			self.server.shutdown()
			self.log("server closed successfully")
		if self.loopThread:
			self.log("loop thread final", self.loopThread, self.loopThread.is_alive())



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
		self.heartbeatThread = threading.Thread(target=self.heartbeatLoop, daemon=True)
		self.loopThread.start()
		self.heartbeatThread.start()

	def run(self):
		"""startup method that begins event loop, handles errors
		for now we just hack an event loop here
		"""
		self.server.serve_forever(poll_interval=0.1)

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
		data["s"] = (self.portId(), self.name)
		data["r"] = wantResponse
		return data

	def send(self, msg:dict, toPort=None):
		"""send a message from this session back to the bridge

		TODO: unsure if there's a point in sending a direct response back
			this way, maybe for confirmation of publish, in case there's an
			error in the export somehow?
			but naively I'd rather catch and filter that as a general incoming message
		"""
		if toPort is None:
			if isinstance(self, DCCIdemSession):
				toPort = self.connectedBridgeId()
			else:
				toPort = self.portId()

		log(self._sessionFileName(), "send to", toPort, msg)

		sock = socket.create_connection(
			("localhost", toPort), timeout=3.0
		)
		self.log("created conn", toPort, sock)


		#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		with sock:
			# if msg["r"]:  # allow longer for response
			# 	sock.settimeout(1.0)
			# 	sock.setblocking(True)
			# else:
			# 	sock.settimeout(1.0)

			# try:
			# 	sock.connect(("localhost", toPort))
			# except ConnectionRefusedError:
			# 	log(self.uuid, f"IDEM ERROR: no listening port on {toPort}, aborting send")
			# 	return
			# except socket.timeout:
			# 	# later consider retrying any 'important' commands?
			# 	log(self.uuid, f"IDEM ERROR: port {toPort} timed out, aborting send")
			# 	return
			libsocket.sendMsg(sock, msg)

			if msg["r"]: # wait for response
				self.log("wait for reply")
				result = libsocket.recvMsg(sock)
				self.log( "got reply", result)
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

	# def _sessionFileName(self)->str:
	# 	return f"{self.uuid}_{self.dccType}_{self._getDCCFileName()}"
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

	def getHeartbeatPorts(self) ->list[int]:
		"""return ports to ping on heartbeat -
		for session, only bridge (if connected)"""
		if not self.connectedBridgeId():
			return []
		return [self.connectedBridgeId()]

	def onHeartbeatTimeout(self, timeoutPorts:list[int]):
		"""connected bridge has timed out, reset this server"""
		if not timeoutPorts:
			return
		bridgeId = timeoutPorts[0]
		if not self.connectedBridgeId():
			return
		self.disconnectBridge(sendCmd=False)

	def handleMessage(self, handler:SlotRequestHandler, msg:dict):
		"""handle incoming messages from bridge"""
		log(self._sessionFileName(), "handle", msg)
		if isinstance(msg, ReplicateDataCmd):
			self.replicatedData = msg["data"]
			return
		if isinstance(msg, ConnectToBridgeCmd):
			self.updateConnectedBridge(msg["s"][0])
			handler.sendResponse(
				self.message({"connectedSession" : self.getSessionIdData()}, # should we have a function to make this response?
				             wantResponse=False)
			)
			return
		if isinstance(msg, DisconnectBridgeCmd): # bridge told this to disconnect
			self.disconnectBridge(sendCmd=False)
		#return super().handleMessage(handler, data)

	def getSessionCamera(self):
		"""return a DCC camera to link to idem,
		updating views simultaneously across programs.
		If this works it'll be so cool"""




