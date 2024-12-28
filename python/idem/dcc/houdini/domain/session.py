from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


from wplib import to, toArr
from idem.dcc.abstract import DCCIdemSession
from idem.dcc.abstract.command import *
import hou
from wph.w3d.data import CameraData
from wph.lib.callback import WpHoudiniCallback

log("LOADED MODULE, cls", CameraData)

class HoudiniIdemSession(DCCIdemSession):
	""""""
	dccType = "houdini"
	cameraCallbackObj : WpHoudiniCallback

	_session : HoudiniIdemSession = None
	@classmethod
	def session(cls)->(None, HoudiniIdemSession):
		return cls._session
	@classmethod
	def getSession(cls, name="houdiniIdem")->HoudiniIdemSession:
		if not cls._session:
			cls._session = cls.bootstrap(name)
		return cls._session

	def getSessionCamera(self)->hou.ObjNode:
		"""
		return a dedicated camera node at "obj/IDEM_CAM"
		"""
		if lookup := hou.node("obj/IDEM_CAM"):
			return lookup
		obj = hou.node("obj")
		cam = obj.createNode("cam", "IDEM_CAM")
		return cam

	def emitCameraData(self, cb: WpHoudiniCallback, node:hou.Node,
	                   *args, **kwargs):
		"""send serialised camera data with matrix of given MObject
		"""
		data = CameraData.gather(node)
		self.send(self.message(SetCameraCmd(data=data)))

	def handleMessage(self, handler:SlotRequestHandler, msg:dict):
		super().handleMessage(handler, msg)
		if isinstance(msg, SetCameraCmd):
			# prevent events triggering infinitely
			self.cameraCallbackObj.pause()
			msg["data"].apply(self.getSessionCamera())
			self.cameraCallbackObj.unpause()

	@classmethod
	def bootstrap(cls, sessionName="houdini")->HoudiniIdemSession:
		"""load up a MayaIdemSession from standing start, set up
		ports, hook up idem camera, sets etc
		"""
		log("houdini bootstrap")

		# adding callbacks for camera
		newSession = cls(name=sessionName)

		node = newSession.getSessionCamera()

		log("cameraData cls", CameraData)
		d = CameraData.gather(node)
		log("gathered", d)

		cameraCallback = WpHoudiniCallback(
			fns=[newSession.emitCameraData],
		)
		cameraCallback.attach(
			node.addEventCallback,
			attachPreArgs=( (hou.nodeEventType.ParmTupleChanged, ),
							)
		)
		newSession.cameraCallbackObj = cameraCallback

		return newSession

	def clear(self):
		super().clear()
		if hasattr(self, "cameraCallbackObj"):
			self.cameraCallbackObj.remove()
		self._session = None
