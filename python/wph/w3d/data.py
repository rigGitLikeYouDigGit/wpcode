from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

#from wp.w3d.data import *
from wp.w3d.data import WpData, CameraData as CameraDataAbstract

from wplib import to, toArr
"""maya-side implementations of the various data types"""

import hou

class HoudiniData(WpData):
	pass

class CameraData(HoudiniData, CameraDataAbstract):

	def apply(self:CameraData, node:hou.ObjNode):
		node.setWorldTransform(hou.Matrix4(self["matrix"]))
		pass

	@classmethod
	def gather(cls, node:hou.ObjNode):
		mat : hou.Matrix4 = node.worldTransform()
		return cls(matrix=mat.asTupleOfTuples())
		pass

