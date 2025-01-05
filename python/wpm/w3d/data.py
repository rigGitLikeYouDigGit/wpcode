from __future__ import annotations
import types, typing as T
import pprint

import numpy as np

from wplib import log

from wp.w3d.data import WpData, CameraData as CameraDataAbstract

from wplib import to
from wpm import cmds, om, WN
"""maya-side implementations of the various data types"""

class MayaData(WpData):
	pass

class CameraData(MayaData, CameraDataAbstract):

	def apply(self:CameraData, node:WN.Camera):
		#log("apply", self, node, type(node))
		node = node.shape()
		node.tf().setWorldMatrix(np.array(self["matrix"]))
		pass

	@classmethod
	def gather(cls, node:WN.Camera):

		mat = node.tf().worldMatrix()
		return cls(matrix=mat)
		pass

