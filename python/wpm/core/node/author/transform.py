from __future__ import annotations
import typing as T

from ..gen.transform import Transform as GenTransform
import numpy as np
from wpm import cmds, om, WN, to, arr

class Transform(GenTransform):
	""" moving nodes around in a more fluid way than walls
	of api calls"""
	MFn : om.MFnTransform
	clsApiType = om.MFn.kTransform

	def localMatrix(self)->om.MMatrix:
		return self.dagLocalMatrix_.value()
	def worldMatrix(self)->om.MMatrix:
		return self.worldMatrix_.value()

	def setLocalMatrix(self, mat):
		mat = to(mat, om.MTransformationMatrix)
		self.MFn.setTransformation(mat, om.MSpace.kObject)
	def setWorldMatrix(self, mat):
		mat = to(mat, om.MTransformationMatrix)
		self.MFn.setTransformation(mat, om.MSpace.kWorld)

	def localPos(self)->om.MVector:
		return self.MFn.translation(om.MSpace.kObject)
	def worldPos(self)->om.MVector:
		return self.MFn.translation(om.MSpace.kWorld)

	def setLocalPos(self, v):
		v = to(v, om.MVector)
		self.MFn.setTranslation(v, om.MSpace.kObject)
	def setWorldPos(self, v):
		v = to(v, om.MVector)
		self.MFn.setTranslation(v, om.MSpace.kWorld)