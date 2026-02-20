from __future__ import annotations
import types, typing as T
"""essential code for maya workflows - should not be lightly edited"""

import sys

_canImport = False
try:
	# import wrapped maya modules
	from .patch import cmds, om, omr, oma, omui
	_canImport = True
except (ImportError, AttributeError) as e:
	print("error importing maya modules - may not be running maya")

if _canImport:
	from .api import (
		MObjectRegister, apiTypeMap, apiTypeCodeMap, apiCodeNameMap, apiTypeDataMap,
		mfnDataConstantTypeMap,
		getMFn, asMFn, getCache, getMObject, getMDagPath, getMFnType, toMFnDag, toMFnDep, toMFnMesh,
		toMFnCurve, toMFnSurface, toMFnTransform,
		isDag, isShape, isTransform,
		classConstantValueToNameMap,
		MObjectSet, listMObjects)


	from .node import (
		WN
	)
	#from . import WN

	from .plugtree import Plug

	from .scene import getSceneGlobals, setupGlobals
	from wpm.core.callbackowner import CallbackOwner

	from . import plug
	from .plug import getMPlug, use


"""

can we make our api package threadsafe somehow?


if a thread calls "om.MDagNode.setParent()", can the "om." go through a
check to see if maya is currently executing a callback, or busy?
and if so wait until it isn't?


there are still quite a few areas to develop:

PLUGS : 
	- setting and retrieving correctly typed values from MPlugs
	- check typing validity of nodes
	- descriptors for __set__, rshift connection operators
	- pathable syntax for slicing
	- broadcasting 
	
NODES :
	- got generation working in principle
	

WN() raw errors
WN("nodeThatDoesn'tExist") errors?
WN("myExistingNode") returns best-fitting wrapper instance on that node 


"""


