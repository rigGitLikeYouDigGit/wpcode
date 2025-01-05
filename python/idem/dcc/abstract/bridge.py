from __future__ import annotations

import os
import socketserver
import threading
import types, typing as T
import pprint
from dataclasses import dataclass

from idem.dcc.abstract.session import DataFileServer, SlotRequestHandler
from idem.dcc.abstract.command import *

from wplib import log


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

	for some reason I was starting a separate dedicated server here for
	each linked bridge session - seems unnecessary

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
			"connected" : {} # type:dict[str, int]
		}
		self.writeSessionFileData(self.liveData)

	def getHeartbeatPorts(self) ->list[int]:
		self.log("get heartbeat ports", self.linkedSessions)
		return list(self.linkedSessions.keys())

	def onHeartbeatTimeout(self, timeoutPorts:list[int]):
		for i in timeoutPorts:
			self.disconnectSocket(i, sendCmd=False)

	def connectToSocket(self, portId:int,
	                    sendCmd=True):
		"""create a new server listening on the given child socket

		called on BRIDGE to drive CHILD

		"""

		self.linkedSessions[portId] = ChildSessionData(
			portId, None, serverThread=None)

		# ask server for some other information
		# driving CHILD from BRIDGE
		if sendCmd:
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
				self.liveData["connected"][str(portId)] = portId
				self.writeSessionFileData(self.liveData)
			return

		else: # driving BRIDGE from CHILD
			# we assume the outer scope already has a response ready
			self.liveData["connected"][str(portId)] = portId
			self.writeSessionFileData(self.liveData)
			pass

	def disconnectSocket(self, portId:int, sendCmd=False):
		"""remove connection to the given port -
		shutdown listening server
		halt listening thread
		remove from connections"""

		self.log("linked sessions", self.linkedSessions)
		self.linkedSessions.pop(portId)
		self.liveData["connected"].pop(str(portId), None) # todo:scrap livedata
		self.writeSessionFileData(self.liveData)

		if sendCmd:
			self.send(self.message(DisconnectBridgeCmd(), wantResponse=False),
			          toPort=portId)


	def send(self, msg:dict, toPort=None, notToPort=None):
		if toPort is None:
			for k, server in self.linkedSessions.items():
				# can't get response from general message
				if server.port == notToPort:
					continue
				msg["r"] = False
				self.send(msg, toPort=k)
			return
		return super().send(msg, toPort)

	def handleMessage(self, handler:SlotRequestHandler, msg:(IdemCmd, dict), *args,
	                      fromPort=0, **kwargs):
		"""if fromPort is None, probably came from
		this bridge session"""
		self.log("handle :", msg)
		self.log(type(msg), isinstance(msg, DisconnectSessionCmd))
		if isinstance(msg, ConnectToSessionCmd): # sent by session
			# connect to the port that sent this message
			self.connectToSocket(msg["s"][0], sendCmd=False)
			handler.sendResponse({"connectedSession" : self.getSessionIdData()})
			return
		if isinstance(msg, DisconnectSessionCmd): # sent by session
			self.log("disconnecting from msg")
			self.disconnectSocket(msg["s"][0], sendCmd=False)
			self.log("disconnected session from bridge", msg["s"][0])
			self.log(self.linkedSessions)
			return

		if isinstance(msg, ReplicateDataCmd):
			self.replicatedData = msg["data"]

		# by default echo each command received by bridge to all sessions,
		# except its own sender
		self.send(msg, notToPort=msg["s"][0])
	
	def clear(self):
		"""when bridge shuts down, disconnect all connected sessions"""
		self.send(self.message(DisconnectBridgeCmd, ))
		super().clear()
