
from __future__ import annotations
import typing as T

import numpy as np
from wpm import cmds, om, WN, arr, getMObject
from wplib.totype import to
from ..gen.dagNode import DagNode as GenDagNode

if T.TYPE_CHECKING:
	# add any extra imports
	if T.TYPE_CHECKING:
		from ..author import Catalogue
		GenDagNode = Catalogue.DagNode


class DagNode(GenDagNode):
	clsIsDag = True
	MFn : om.MFnDagNode

	def setParentKeepLocal(self, other:WN):
		"""reparent while preserving local attributes,
		causing transform to move physically in worldspace"""
		origParent = None
		if self.MFn.parentCount():
			origParent = self.MFn.parent(0)
		otherMFn = om.MFnDagNode(getMObject(other))
		otherMFn.addChild(self.object())
		if origParent:
			om.MFnDagNode(origParent).removeChild(self.object())




