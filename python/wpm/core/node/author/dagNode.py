
from __future__ import annotations
import typing as T

import numpy as np
from wpm import cmds, om, WN, arr, getMObject
from wplib.totype import to, coerce
from ..gen.dagNode import DagNode as GenDagNode

if T.TYPE_CHECKING:
	# add any extra imports
	if T.TYPE_CHECKING:
		from ..author import Catalogue
		GenDagNode = Catalogue.DagNode


class DagNode(GenDagNode):
	clsIsDag = True
	MFn : om.MFnDagNode

	@coerce
	def setParentKeepLocal(self, other:om.MObject):
		"""reparent while preserving local attributes,
		causing transform to move physically in worldspace"""
		origParent = None
		if self.MFn.parentCount():
			origParent = self.MFn.parent(0)
		otherMFn = om.MFnDagNode(other)
		otherMFn.addChild(self.object())
		if origParent:
			om.MFnDagNode(origParent).removeChild(self.object())

	# alias to match the api
	setParent = setParentKeepLocal

	@coerce
	def addChild(self, other:om.MObject):
		otherMFn = om.MFnDagNode(other)
		otherParents = [otherMFn.parent(i)
		                for i in range(otherMFn.parentCount())]
		self.MFn.addChild(otherMFn)
		for p in otherParents:
			om.MFnDagNode(p).removeChild(other)


