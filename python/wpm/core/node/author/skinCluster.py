from __future__ import annotations
import typing as T

from ..gen.skinCluster import SkinCluster as GenSkinCluster

import numpy as np

from wplib.object import IndexScope, IndexScopeGroup

from wpm import cmds, om, WN, arr, oma

from wpm.lib import skin # unsure if illegal

if T.TYPE_CHECKING:
	from ...node.base import Plug


class SkinCluster(GenSkinCluster):
	""" builtin methods for getting history, future ,
	origshapes etc
	"""
	MFn: oma.MFnSkinCluster
	clsApiType = om.MFn.kSkinClusterFilter

	def nInfluences(self)->int:
		return len(self.bindPreMatrix_)

	def bindPreMatrices(self)->np.ndarray[N, 4, 4]:
		matArr = np.zeros((self.nInfluences(), 4, 4), dtype=float)
		for i in range(self.nInfluences()):
			matArr[i] = self.bindPreMatrix_[i].asMatrix()
		return matArr





