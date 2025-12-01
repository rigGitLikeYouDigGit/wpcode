

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
BaseShadingSwitch = retriever.getNodeCls("BaseShadingSwitch")
assert BaseShadingSwitch
if T.TYPE_CHECKING:
	from .. import BaseShadingSwitch

# add node doc



# region plug type defs
class DefaultPlug(Plug):
	node : SingleShadingSwitch = None
	pass
class InShapePlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : SingleShadingSwitch = None
	pass
class InSinglePlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : SingleShadingSwitch = None
	pass
class InputPlug(Plug):
	inShape_ : InShapePlug = PlugDescriptor("inShape")
	is_ : InShapePlug = PlugDescriptor("inShape")
	inSingle_ : InSinglePlug = PlugDescriptor("inSingle")
	it_ : InSinglePlug = PlugDescriptor("inSingle")
	node : SingleShadingSwitch = None
	pass
class OutputPlug(Plug):
	node : SingleShadingSwitch = None
	pass
# endregion


# define node class
class SingleShadingSwitch(BaseShadingSwitch):
	default_ : DefaultPlug = PlugDescriptor("default")
	inShape_ : InShapePlug = PlugDescriptor("inShape")
	inSingle_ : InSinglePlug = PlugDescriptor("inSingle")
	input_ : InputPlug = PlugDescriptor("input")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "singleShadingSwitch"
	apiTypeInt = 618
	apiTypeStr = "kSingleShadingSwitch"
	typeIdInt = 1398229041
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["default", "inShape", "inSingle", "input", "output"]
	nodeLeafPlugs = ["default", "input", "output"]
	pass

