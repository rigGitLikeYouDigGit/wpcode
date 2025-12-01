

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyMoveVertex = retriever.getNodeCls("PolyMoveVertex")
assert PolyMoveVertex
if T.TYPE_CHECKING:
	from .. import PolyMoveVertex

# add node doc



# region plug type defs
class LocalCenterPlug(Plug):
	node : PolyMoveEdge = None
	pass
class LocalRotateXPlug(Plug):
	parent : LocalRotatePlug = PlugDescriptor("localRotate")
	node : PolyMoveEdge = None
	pass
class LocalRotateYPlug(Plug):
	parent : LocalRotatePlug = PlugDescriptor("localRotate")
	node : PolyMoveEdge = None
	pass
class LocalRotateZPlug(Plug):
	parent : LocalRotatePlug = PlugDescriptor("localRotate")
	node : PolyMoveEdge = None
	pass
class LocalRotatePlug(Plug):
	localRotateX_ : LocalRotateXPlug = PlugDescriptor("localRotateX")
	lrx_ : LocalRotateXPlug = PlugDescriptor("localRotateX")
	localRotateY_ : LocalRotateYPlug = PlugDescriptor("localRotateY")
	lry_ : LocalRotateYPlug = PlugDescriptor("localRotateY")
	localRotateZ_ : LocalRotateZPlug = PlugDescriptor("localRotateZ")
	lrz_ : LocalRotateZPlug = PlugDescriptor("localRotateZ")
	node : PolyMoveEdge = None
	pass
class LocalScaleXPlug(Plug):
	parent : LocalScalePlug = PlugDescriptor("localScale")
	node : PolyMoveEdge = None
	pass
class LocalScaleYPlug(Plug):
	parent : LocalScalePlug = PlugDescriptor("localScale")
	node : PolyMoveEdge = None
	pass
class LocalScaleZPlug(Plug):
	parent : LocalScalePlug = PlugDescriptor("localScale")
	node : PolyMoveEdge = None
	pass
class LocalScalePlug(Plug):
	localScaleX_ : LocalScaleXPlug = PlugDescriptor("localScaleX")
	lsx_ : LocalScaleXPlug = PlugDescriptor("localScaleX")
	localScaleY_ : LocalScaleYPlug = PlugDescriptor("localScaleY")
	lsy_ : LocalScaleYPlug = PlugDescriptor("localScaleY")
	localScaleZ_ : LocalScaleZPlug = PlugDescriptor("localScaleZ")
	lsz_ : LocalScaleZPlug = PlugDescriptor("localScaleZ")
	node : PolyMoveEdge = None
	pass
# endregion


# define node class
class PolyMoveEdge(PolyMoveVertex):
	localCenter_ : LocalCenterPlug = PlugDescriptor("localCenter")
	localRotateX_ : LocalRotateXPlug = PlugDescriptor("localRotateX")
	localRotateY_ : LocalRotateYPlug = PlugDescriptor("localRotateY")
	localRotateZ_ : LocalRotateZPlug = PlugDescriptor("localRotateZ")
	localRotate_ : LocalRotatePlug = PlugDescriptor("localRotate")
	localScaleX_ : LocalScaleXPlug = PlugDescriptor("localScaleX")
	localScaleY_ : LocalScaleYPlug = PlugDescriptor("localScaleY")
	localScaleZ_ : LocalScaleZPlug = PlugDescriptor("localScaleZ")
	localScale_ : LocalScalePlug = PlugDescriptor("localScale")

	# node attributes

	typeName = "polyMoveEdge"
	apiTypeInt = 418
	apiTypeStr = "kPolyMoveEdge"
	typeIdInt = 1347243845
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["localCenter", "localRotateX", "localRotateY", "localRotateZ", "localRotate", "localScaleX", "localScaleY", "localScaleZ", "localScale"]
	nodeLeafPlugs = ["localCenter", "localRotate", "localScale"]
	pass

