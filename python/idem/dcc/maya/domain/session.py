from __future__ import annotations
import types, typing as T
import pprint

from idem.dcc.abstract.session import SlotRequestHandler
from wplib import log



from wplib import to, toArr
from wpm import cmds, om, WN

from idem.dcc.abstract import DCCIdemSession
from idem.dcc.abstract.command import *
from wpm.w3d.data import CameraData
from wpm.lib.callback import WpMayaCallback

class MayaIdemSession(DCCIdemSession):
	""""""

	cameraCallbackObj : WpMayaCallback

	if T.TYPE_CHECKING:
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

	def emitCameraData(self, cb: WpMayaCallback, mobj: om.MObject, userData=None, *args):
		"""send serialised camera data with matrix of given MObject
		"""
		if not self.connectedBridgeId():
			return
		log("emitCameraData", cb, mobj, userData, *args)
		mfn = om.MFnTransform(mobj)
		# todo: obvs replace this with proper library functions from plugs
		mat = om.MFnMatrixData(mfn.findPlug("worldMatrix", 0).elementByLogicalIndex(0).asMObject()).matrix()
		data = CameraData(matrix=toArr(mat).tolist())

		self.send(self.message(SetCameraCmd(data=data)), toPort=self.connectedBridgeId())
		#print("emitted", data)

	def handleMessage(self, handler:SlotRequestHandler, msg:dict):
		super().handleMessage(handler, msg)
		if isinstance(msg, SetCameraCmd):
			# prevent events triggering infinitely
			self.cameraCallbackObj.pause()
			#msg["data"].apply(self.getSessionCamera())
			self.log("idem cam", self.getSessionCamera(), type(self.getSessionCamera()))
			CameraData.apply(msg["data"], self.getSessionCamera())
			self.cameraCallbackObj.unpause()

	@classmethod
	def bootstrap(cls, sessionName="maya")->MayaIdemSession:
		"""load up a MayaIdemSession from standing start, set up
		ports, hook up idem camera, sets etc
		"""
		newSession = super().bootstrap(sessionName)

		stayAliveFn = lambda *a, **kw : cls.session() is not None

		cameraCallback = WpMayaCallback(
			fns=[newSession.emitCameraData],
			stayAliveFn=stayAliveFn
		)
		log("attaching to camera", newSession.getSessionCamera(), type(newSession.getSessionCamera()))
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



