
from __future__ import annotations
import typing as T

import numpy as np
from wpm import cmds, om, WN, to, arr, getMObject
from ..gen.dagNode import DagNode as GenDagNode


class DagNode(GenDagNode):
	clsIsDag = True
	MFn : om.MFnDagNode

	def setParentKeepLocal(self, other:WN):
		"""reparent while preserving local attributes,
		causing transform to move physically in worldspace"""
		otherMFn = om.MFnDagNode(getMObject(other))
		otherMFn.addChild(self.object())



