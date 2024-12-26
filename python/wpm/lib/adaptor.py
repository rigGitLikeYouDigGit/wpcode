from __future__ import annotations
import types, typing as T
import pprint

import numpy as np

from wplib import log


from wplib.maths import NPArrayLike, arr, to, ToType
from wpm import cmds, om
import maya.api.OpenMaya as om_

mMatrixEdge = ToType(
	fromTypes=(om.MMatrix, om.MFloatMatrix,
	            om_.MMatrix, om_.MFloatMatrix),
	toTypes=(np.ndarray,),
	convertFn=lambda v, t, **kw : t([
			[v[0], v[1], v[2], v[3]],
			[v[4], v[5], v[6], v[7]],
			[v[8], v[9], v[10], v[11]],
			[v[12], v[13], v[14], v[15]]
		]),
	backFn=lambda v, t, **kw : t(v)
)

mVectorEdge = ToType(
	fromTypes=(om.MVector, om.MFloatVector,
	            om_.MVector, om_.MFloatVector),
	toTypes=(np.ndarray,),
)





