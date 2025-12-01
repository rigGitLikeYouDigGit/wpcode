

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
class DirectionXPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : ProjectCurve = None
	pass
class DirectionYPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : ProjectCurve = None
	pass
class DirectionZPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : ProjectCurve = None
	pass
class DirectionPlug(Plug):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	dx_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	dy_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	dz_ : DirectionZPlug = PlugDescriptor("directionZ")
	node : ProjectCurve = None
	pass
class InputCurvePlug(Plug):
	node : ProjectCurve = None
	pass
class InputSurfacePlug(Plug):
	node : ProjectCurve = None
	pass
class OutputCurvePlug(Plug):
	node : ProjectCurve = None
	pass
class TolerancePlug(Plug):
	node : ProjectCurve = None
	pass
class UseNormalPlug(Plug):
	node : ProjectCurve = None
	pass
# endregion


# define node class
class ProjectCurve(AbstractBaseCreate):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")
	useNormal_ : UseNormalPlug = PlugDescriptor("useNormal")

	# node attributes

	typeName = "projectCurve"
	apiTypeInt = 87
	apiTypeStr = "kProjectCurve"
	typeIdInt = 1313882962
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["directionX", "directionY", "directionZ", "direction", "inputCurve", "inputSurface", "outputCurve", "tolerance", "useNormal"]
	nodeLeafPlugs = ["direction", "inputCurve", "inputSurface", "outputCurve", "tolerance", "useNormal"]
	pass

