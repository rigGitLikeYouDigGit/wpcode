from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from idem.dcc.abstract import DCCIdemSession

from wpm import cmds, om, WN
from wpm.w3d.data import CameraData

class MayaIdemSession(DCCIdemSession):
	""""""

	def getSessionCamera(self)->WN:
		"""for now, just return the persp camera -
		should probably make an "idem_CAM" node
		later on
		"""
		return WN("persp")


	@classmethod
	def bootstrap(cls)->MayaIdemSession:
		"""load up a MayaIdemSession from standing start, set up
		ports, hook up idem camera, sets etc
		"""

		# adding callbacks for camera

		#### MEventMessage cameraChange doesn't do anything
		# cbID = om.MEventMessage.addEventCallback(
		# 	"cameraChange",
		# 	onCameraChange
		# )





