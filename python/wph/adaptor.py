from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import numpy as np
from wplib import to, toArr
from wplib.object import ToType

import hou

"""
np.array( houdiniMatrix4 ) works fine :)
hou.Matrix( npArray ) crashes houdini :)
"""

houMatEdge = ToType(
	fromTypes=(hou.Matrix3, hou.Matrix4),
	toTypes=(np.ndarray, ),
	convertFn=lambda v, t, **kwargs : np.array(v),
	backFn=lambda v, t, **kwargs : t(v.tolist())
)

houVecEdge = ToType(
	fromTypes=(hou.Vector3, hou.Vector4),
	toTypes=(np.ndarray, ),
	convertFn=lambda v, t, **kwargs : np.array(v),
	backFn=lambda v, t, **kwargs : t(v.tolist())
)
