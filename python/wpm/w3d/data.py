from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wp.w3d.data import *

from wpm import cmds, om, WN
"""maya-side implementations of the various data types"""


class CameraData(CameraData):

	def apply(self, node:WN.Camera):
		node = node.shape
		om.MFnTransform(node.transform.object()).setTransformation(
			om.MTransformationMatrix()
		)
		pass

	@classmethod
	def gather(cls, node:WN.Camera):
		pass

