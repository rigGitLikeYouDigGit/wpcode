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
		return self.dagLocalMatrix_()
	def worldMatrix(self)->om.MMatrix:
		return self.worldMatrix_[0]()

	def setLocalMatrix(self, mat):
		mat = to(mat, om.MTransformationMatrix)
		self.MFn.setTransformation(mat)
	def setWorldMatrix(self, mat):
		"""inverse of parent * mat for local"""
		if not self.parent():
			mat = to(mat, om.MTransformationMatrix)
			self.MFn.setTransformation(mat)
			return

		parentMat = om.MFnTransform(self.MFn.parent(0)).transformationMatrix()
		mat = to(mat, om.MMatrix)
		self.setLocalMatrix(parentMat.inverse() * mat)

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

	@property
	def localIn(self) -> Plug:
		return self.matrix_
	@property
	def localOut(self) -> Plug:
		return self.matrix_

	@property
	def worldIn(self) -> Plug:
		return self.worldMatrix_[0]
	@property
	def worldOut(self) -> Plug:
		return self.worldMatrix_[0]

	def setParentKeepWorld(self, other:WN, updateOffsetParentMat=False):
		"""reparent keeping worldspace position, setting new local
		transform attributes on this node
		if updateOffsetParentMat, the offsetParentMatrix of this node will
		receive the full transform, and both world position and local attributes
		of this node will be preserved (as long as offsetParent is not
		connected in the graph)
		"""
		worldMat = self.worldMatrix()
		self.setParentKeepLocal(other)
		self.setWorldMatrix(worldMat)