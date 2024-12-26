from __future__ import annotations
import types, typing as T
import pprint

import numpy as np

from wplib import log


from wplib.maths import NPArrayLike, arr, to, ToType, toArr
from wpm.core import cmds, om
import maya.api.OpenMaya as om_

mMatrixEdge = ToType(
	fromTypes=(om.MMatrix, om.MFloatMatrix,
	            om_.MMatrix, om_.MFloatMatrix),
	toTypes=(np.ndarray,),
	convertFn=lambda v, t, **kw : np.array([
			[v[0], v[1], v[2], v[3]],
			[v[4], v[5], v[6], v[7]],
			[v[8], v[9], v[10], v[11]],
			[v[12], v[13], v[14], v[15]]
		]),
	backFn=lambda v, t, **kw : t(v)
)
log("made matrix edge")
log(ToType.typeGraph)
def mTransformationMatrixToMMatrixFn(v:om.MTransformationMatrix,
                                     t:type[om.MMatrix],
                                     **kwargs)->om.MMatrix:
	return v.asMatrix()

mTransformationMatrixEdge = ToType(
	fromTypes=(om.MTransformationMatrix, ),
	toTypes=(om.MMatrix, ),
	convertFn=mTransformationMatrixToMMatrixFn,
	backFn=lambda v, t, **kwargs: t(v)
)

mVectorEdge = ToType(
	fromTypes=(om.MVector, om.MFloatVector,
	            om_.MVector, om_.MFloatVector),
	toTypes=(np.ndarray,),
	convertFn=lambda v, t, **kwargs: t(v),
	backFn=lambda v, t, **kwargs : t(v[:3])
)

mPointEdge = ToType(
	fromTypes=(om.MPoint, om.MFloatPoint,
	            om_.MPoint, om_.MFloatPoint),
	toTypes=(np.ndarray,),
	convertFn=lambda v, t, **kwargs: np.array(v),
	backFn=lambda v, t, **kwargs : t(*v)
)

mMatVectorEdge = ToType(
	fromTypes=(om.MMatrix, om.MFloatMatrix,
	           om_.MMatrix, om_.MFloatMatrix),
	toTypes=(om.MVector, om.MFloatVector,
	            om_.MVector, om_.MFloatVector),
	convertFn=lambda v, t, **kwargs : t(v[12], v[13], v[14]),
	backFn=lambda v, t, **kwargs : t([
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		v[0], v[1], v[2], 0
	])
)



