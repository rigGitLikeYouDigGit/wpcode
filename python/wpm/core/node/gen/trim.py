

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs
class InputCurvePlug(Plug):
	node : Trim = None
	pass
class InputSurfacePlug(Plug):
	node : Trim = None
	pass
class LocatorUPlug(Plug):
	node : Trim = None
	pass
class LocatorVPlug(Plug):
	node : Trim = None
	pass
class OutputSurfacePlug(Plug):
	node : Trim = None
	pass
class SelectedPlug(Plug):
	node : Trim = None
	pass
class ShouldBeLastPlug(Plug):
	node : Trim = None
	pass
class ShrinkPlug(Plug):
	node : Trim = None
	pass
class SplitSurfacePlug(Plug):
	node : Trim = None
	pass
class TolerancePlug(Plug):
	node : Trim = None
	pass
class UsedCurvesPlug(Plug):
	node : Trim = None
	pass
# endregion


# define node class
class Trim(AbstractBaseCreate):
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	locatorU_ : LocatorUPlug = PlugDescriptor("locatorU")
	locatorV_ : LocatorVPlug = PlugDescriptor("locatorV")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	selected_ : SelectedPlug = PlugDescriptor("selected")
	shouldBeLast_ : ShouldBeLastPlug = PlugDescriptor("shouldBeLast")
	shrink_ : ShrinkPlug = PlugDescriptor("shrink")
	splitSurface_ : SplitSurfacePlug = PlugDescriptor("splitSurface")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")
	usedCurves_ : UsedCurvesPlug = PlugDescriptor("usedCurves")

	# node attributes

	typeName = "trim"
	apiTypeInt = 105
	apiTypeStr = "kTrim"
	typeIdInt = 1314148941
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["inputCurve", "inputSurface", "locatorU", "locatorV", "outputSurface", "selected", "shouldBeLast", "shrink", "splitSurface", "tolerance", "usedCurves"]
	nodeLeafPlugs = ["inputCurve", "inputSurface", "locatorU", "locatorV", "outputSurface", "selected", "shouldBeLast", "shrink", "splitSurface", "tolerance", "usedCurves"]
	pass

