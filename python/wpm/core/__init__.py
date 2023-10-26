
"""essential code for maya workflows - should not be lightly edited"""

import sys


# import wrapped maya modules
from .patch import cmds, om, omr, oma, omui

from .api import (
	MObjectRegister, apiTypeMap, apiTypeCodeMap, apiCodeNameMap, apiTypeDataMap,
	mfnDataConstantTypeMap,
	getMFn, asMFn, getCache, getMObject, getMFnType, toMFnDag, toMFnDep, toMFnMesh,
	toMFnCurve, toMFnSurface, toMFnTransform,
	isDag, isShape, isTransform,
	classConstantValueToNameMap,
	MObjectSet, listMObjects)


from .node import (
	WN, createWN
)

from .plug import PlugTree

from .scene import getSceneGlobals, setupGlobals
from wpm.core.callbackowner import CallbackOwner

from . import attr