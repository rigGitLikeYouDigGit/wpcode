

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
class DefComp1Plug(Plug):
	parent : DefaultPlug = PlugDescriptor("default")
	node : DoubleShadingSwitch = None
	pass
class DefComp2Plug(Plug):
	parent : DefaultPlug = PlugDescriptor("default")
	node : DoubleShadingSwitch = None
	pass
class DefaultPlug(Plug):
	defComp1_ : DefComp1Plug = PlugDescriptor("defComp1")
	dc1_ : DefComp1Plug = PlugDescriptor("defComp1")
	defComp2_ : DefComp2Plug = PlugDescriptor("defComp2")
	dc2_ : DefComp2Plug = PlugDescriptor("defComp2")
	node : DoubleShadingSwitch = None
	pass
class InComp1Plug(Plug):
	parent : InDoublePlug = PlugDescriptor("inDouble")
	node : DoubleShadingSwitch = None
	pass
class InComp2Plug(Plug):
	parent : InDoublePlug = PlugDescriptor("inDouble")
	node : DoubleShadingSwitch = None
	pass
class InDoublePlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	inComp1_ : InComp1Plug = PlugDescriptor("inComp1")
	ic1_ : InComp1Plug = PlugDescriptor("inComp1")
	inComp2_ : InComp2Plug = PlugDescriptor("inComp2")
	ic2_ : InComp2Plug = PlugDescriptor("inComp2")
	node : DoubleShadingSwitch = None
	pass
class InShapePlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : DoubleShadingSwitch = None
	pass
class InputPlug(Plug):
	inDouble_ : InDoublePlug = PlugDescriptor("inDouble")
	idl_ : InDoublePlug = PlugDescriptor("inDouble")
	inShape_ : InShapePlug = PlugDescriptor("inShape")
	is_ : InShapePlug = PlugDescriptor("inShape")
	node : DoubleShadingSwitch = None
	pass
class OutComp1Plug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : DoubleShadingSwitch = None
	pass
class OutComp2Plug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : DoubleShadingSwitch = None
	pass
class OutputPlug(Plug):
	outComp1_ : OutComp1Plug = PlugDescriptor("outComp1")
	oc1_ : OutComp1Plug = PlugDescriptor("outComp1")
	outComp2_ : OutComp2Plug = PlugDescriptor("outComp2")
	oc2_ : OutComp2Plug = PlugDescriptor("outComp2")
	node : DoubleShadingSwitch = None
	pass
# endregion


# define node class
class DoubleShadingSwitch(BaseShadingSwitch):
	defComp1_ : DefComp1Plug = PlugDescriptor("defComp1")
	defComp2_ : DefComp2Plug = PlugDescriptor("defComp2")
	default_ : DefaultPlug = PlugDescriptor("default")
	inComp1_ : InComp1Plug = PlugDescriptor("inComp1")
	inComp2_ : InComp2Plug = PlugDescriptor("inComp2")
	inDouble_ : InDoublePlug = PlugDescriptor("inDouble")
	inShape_ : InShapePlug = PlugDescriptor("inShape")
	input_ : InputPlug = PlugDescriptor("input")
	outComp1_ : OutComp1Plug = PlugDescriptor("outComp1")
	outComp2_ : OutComp2Plug = PlugDescriptor("outComp2")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "doubleShadingSwitch"
	apiTypeInt = 619
	apiTypeStr = "kDoubleShadingSwitch"
	typeIdInt = 1398229042
	MFnCls = om.MFnDependencyNode
	pass

