
"""essential code for maya workflows - should not be lightly edited"""

import sys


# import wrapped maya modules
from .patch import cmds, om, omr, oma, omui

from .api import (
	MObjectRegister, apiTypeMap, apiTypeCodeMap, apiCodeNameMap, apiTypeDataMap,
	mfnDataConstantTypeMap,
	toMFn, getCache, toMObject, getMFnType, toMFnDag, toMFnDep, toMFnMesh,
	toMFnCurve, toMFnSurface, toMFnTransform,
	isDag, isShape, isTransform,
	MObjectSet, listMObjects)


from .node import (
	WN, createWN
)

from .scene import getSceneGlobals, setupGlobals