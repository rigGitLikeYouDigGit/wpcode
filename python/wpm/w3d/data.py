from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wp.w3d.data import *

from wpm import cmds, om, WN
"""maya-side implementations of the various data types"""

class MayaData(WpData):
	pass

class CameraData(WpData, CameraData):

	def apply(self:CameraData, node:WN.Camera):
		node = node.shape
		node.transform.setWorldMatrix(self["matrix"])
		pass

	@classmethod
	def gather(cls, node:WN.Camera):
		mat = node.transform.worldMatrix()
		return cls(matrix=mat)
		pass

