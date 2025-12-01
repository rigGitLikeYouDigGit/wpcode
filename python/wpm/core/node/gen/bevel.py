

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class BevelShapeTypePlug(Plug):
	node : Bevel = None
	pass
class CornerTypePlug(Plug):
	node : Bevel = None
	pass
class DepthPlug(Plug):
	node : Bevel = None
	pass
class ExtrudeDepthPlug(Plug):
	node : Bevel = None
	pass
class InputCurvePlug(Plug):
	node : Bevel = None
	pass
class JoinSurfacesPlug(Plug):
	node : Bevel = None
	pass
class NumberOfSidesPlug(Plug):
	node : Bevel = None
	pass
class OutputSurface1Plug(Plug):
	node : Bevel = None
	pass
class OutputSurface2Plug(Plug):
	node : Bevel = None
	pass
class OutputSurface3Plug(Plug):
	node : Bevel = None
	pass
class TolerancePlug(Plug):
	node : Bevel = None
	pass
class UseDirectionCurvePlug(Plug):
	node : Bevel = None
	pass
class WidthPlug(Plug):
	node : Bevel = None
	pass
# endregion


# define node class
class Bevel(AbstractBaseCreate):
	bevelShapeType_ : BevelShapeTypePlug = PlugDescriptor("bevelShapeType")
	cornerType_ : CornerTypePlug = PlugDescriptor("cornerType")
	depth_ : DepthPlug = PlugDescriptor("depth")
	extrudeDepth_ : ExtrudeDepthPlug = PlugDescriptor("extrudeDepth")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	joinSurfaces_ : JoinSurfacesPlug = PlugDescriptor("joinSurfaces")
	numberOfSides_ : NumberOfSidesPlug = PlugDescriptor("numberOfSides")
	outputSurface1_ : OutputSurface1Plug = PlugDescriptor("outputSurface1")
	outputSurface2_ : OutputSurface2Plug = PlugDescriptor("outputSurface2")
	outputSurface3_ : OutputSurface3Plug = PlugDescriptor("outputSurface3")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")
	useDirectionCurve_ : UseDirectionCurvePlug = PlugDescriptor("useDirectionCurve")
	width_ : WidthPlug = PlugDescriptor("width")

	# node attributes

	typeName = "bevel"
	apiTypeInt = 48
	apiTypeStr = "kBevel"
	typeIdInt = 1312970316
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["bevelShapeType", "cornerType", "depth", "extrudeDepth", "inputCurve", "joinSurfaces", "numberOfSides", "outputSurface1", "outputSurface2", "outputSurface3", "tolerance", "useDirectionCurve", "width"]
	nodeLeafPlugs = ["bevelShapeType", "cornerType", "depth", "extrudeDepth", "inputCurve", "joinSurfaces", "numberOfSides", "outputSurface1", "outputSurface2", "outputSurface3", "tolerance", "useDirectionCurve", "width"]
	pass

