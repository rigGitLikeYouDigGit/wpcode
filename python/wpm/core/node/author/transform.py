from __future__ import annotations
import typing as T

from ..gen.transform import Transform as GenTransform
import numpy as np
from wpm import cmds, om, WN, arr, fromArr

class Transform(GenTransform):
	""" moving nodes around in a more fluid way than walls
	of api calls"""
	MFn : om.MFnTransform
	clsApiType = om.MFn.kTransform

	def localMatrix(self)->om.MMatrix:
		return self.dagLocalMatrix_.value()
	def localMatrixArr(self)->np.ndarray:
		return arr(self.localMatrix())
	def localMatrixMTransformation(self)->om.MTransformationMatrix:
		return self.MFn.transformation()
	def worldMatrix(self)->om.MMatrix:
		return self.worldMatrix_.value()

	def setLocalMatrix(self, mat):
		if isinstance()

	def localPos(self)->om.MVector:
		return self.MFn.translation(om.MSpace.kObject)
	def worldPos(self)->om.MVector:
		return self.MFn.translation(om.MSpace.kWorld)
