from __future__ import annotations
import types, typing as T
import pprint

from idem.dcc.abstract.session import SlotRequestHandler
from wplib import log



from wplib import to, toArr
from wpm import cmds, om, WN
from wpm.w3d.data import CameraData
from wpm.lib.callback import WpCallback

from idem.dcc.abstract import DCCIdemSession
from idem.dcc.abstract.command import *

class MayaIdemSession(DCCIdemSession):
	""""""

	cameraCallbackObj : WpCallback

	_session :MayaIdemSession = None
	@classmethod
	def session(cls)->(None, MayaIdemSession):
		return cls._session
	@classmethod
	def getSession(cls, name="mayaIdem")->MayaIdemSession:
		if not cls._session:
			cls._session = cls.bootstrap(name)
		return cls._session

	def getSessionCamera(self)->WN:
		"""for now, just return the persp camera -
		should probably make an "idem_CAM" node
		later on
		"""
		return WN("persp")

	def emitCameraData(self, cb: WpCallback, mobj: om.MObject, userData=None, *args):
		"""send serialised camera data with matrix of given MObject
		"""
		log("emitCameraData", cb, mobj, userData, *args)
		mfn = om.MFnTransform(mobj)
		# todo: obvs replace this with proper library functions from plugs
		mat = om.MFnMatrixData(mfn.findPlug("worldMatrix", 0).elementByLogicalIndex(0).asMObject()).matrix()
		data = CameraData(matrix=toArr(mat).tolist())

		self.send(self.message(SetCameraCmd(data=data)))
		print("emitted", data)

	def handleMessage(self, handler:SlotRequestHandler, msg:dict):
		super().handleMessage(handler, msg)
		if isinstance(msg, SetCameraCmd):
			# prevent events triggering infinitely
			self.cameraCallbackObj.pause()
			msg["data"].apply(self.getSessionCamera())
			self.cameraCallbackObj.unpause()

	@classmethod
	def bootstrap(cls, sessionName="maya")->MayaIdemSession:
		"""load up a MayaIdemSession from standing start, set up
		ports, hook up idem camera, sets etc
		"""
		log("maya bootstrap")

		# adding callbacks for camera
		newSession = cls(name=sessionName)


		cameraCallback = WpCallback(
			fns=[newSession.emitCameraData],
		)
		cameraCallback.attach(
			om.MNodeMessage.addNodeDirtyCallback,
			attachPreArgs=(newSession.getSessionCamera().object(),),
		)
		newSession.cameraCallbackObj = cameraCallback

		return newSession

	def clear(self):
		super().clear()
		self.cameraCallbackObj.remove()
		self._session = None



