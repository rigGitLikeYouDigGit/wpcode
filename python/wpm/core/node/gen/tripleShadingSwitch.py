

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	BaseShadingSwitch = Catalogue.BaseShadingSwitch
else:
	from .. import retriever
	BaseShadingSwitch = retriever.getNodeCls("BaseShadingSwitch")
	assert BaseShadingSwitch

# add node doc



# region plug type defs
class DefComp1Plug(Plug):
	parent : DefaultPlug = PlugDescriptor("default")
	node : TripleShadingSwitch = None
	pass
class DefComp2Plug(Plug):
	parent : DefaultPlug = PlugDescriptor("default")
	node : TripleShadingSwitch = None
	pass
class DefComp3Plug(Plug):
	parent : DefaultPlug = PlugDescriptor("default")
	node : TripleShadingSwitch = None
	pass
class DefaultPlug(Plug):
	defComp1_ : DefComp1Plug = PlugDescriptor("defComp1")
	dc1_ : DefComp1Plug = PlugDescriptor("defComp1")
	defComp2_ : DefComp2Plug = PlugDescriptor("defComp2")
	dc2_ : DefComp2Plug = PlugDescriptor("defComp2")
	defComp3_ : DefComp3Plug = PlugDescriptor("defComp3")
	dc3_ : DefComp3Plug = PlugDescriptor("defComp3")
	node : TripleShadingSwitch = None
	pass
class InComp1Plug(Plug):
	parent : InTriplePlug = PlugDescriptor("inTriple")
	node : TripleShadingSwitch = None
	pass
class InComp2Plug(Plug):
	parent : InTriplePlug = PlugDescriptor("inTriple")
	node : TripleShadingSwitch = None
	pass
class InComp3Plug(Plug):
	parent : InTriplePlug = PlugDescriptor("inTriple")
	node : TripleShadingSwitch = None
	pass
class InShapePlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : TripleShadingSwitch = None
	pass
class InTriplePlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	inComp1_ : InComp1Plug = PlugDescriptor("inComp1")
	ic1_ : InComp1Plug = PlugDescriptor("inComp1")
	inComp2_ : InComp2Plug = PlugDescriptor("inComp2")
	ic2_ : InComp2Plug = PlugDescriptor("inComp2")
	inComp3_ : InComp3Plug = PlugDescriptor("inComp3")
	ic3_ : InComp3Plug = PlugDescriptor("inComp3")
	node : TripleShadingSwitch = None
	pass
class InputPlug(Plug):
	inShape_ : InShapePlug = PlugDescriptor("inShape")
	is_ : InShapePlug = PlugDescriptor("inShape")
	inTriple_ : InTriplePlug = PlugDescriptor("inTriple")
	it_ : InTriplePlug = PlugDescriptor("inTriple")
	node : TripleShadingSwitch = None
	pass
class OutComp1Plug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : TripleShadingSwitch = None
	pass
class OutComp2Plug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : TripleShadingSwitch = None
	pass
class OutComp3Plug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : TripleShadingSwitch = None
	pass
class OutputPlug(Plug):
	outComp1_ : OutComp1Plug = PlugDescriptor("outComp1")
	oc1_ : OutComp1Plug = PlugDescriptor("outComp1")
	outComp2_ : OutComp2Plug = PlugDescriptor("outComp2")
	oc2_ : OutComp2Plug = PlugDescriptor("outComp2")
	outComp3_ : OutComp3Plug = PlugDescriptor("outComp3")
	oc3_ : OutComp3Plug = PlugDescriptor("outComp3")
	node : TripleShadingSwitch = None
	pass
# endregion


# define node class
class TripleShadingSwitch(BaseShadingSwitch):
	defComp1_ : DefComp1Plug = PlugDescriptor("defComp1")
	defComp2_ : DefComp2Plug = PlugDescriptor("defComp2")
	defComp3_ : DefComp3Plug = PlugDescriptor("defComp3")
	default_ : DefaultPlug = PlugDescriptor("default")
	inComp1_ : InComp1Plug = PlugDescriptor("inComp1")
	inComp2_ : InComp2Plug = PlugDescriptor("inComp2")
	inComp3_ : InComp3Plug = PlugDescriptor("inComp3")
	inShape_ : InShapePlug = PlugDescriptor("inShape")
	inTriple_ : InTriplePlug = PlugDescriptor("inTriple")
	input_ : InputPlug = PlugDescriptor("input")
	outComp1_ : OutComp1Plug = PlugDescriptor("outComp1")
	outComp2_ : OutComp2Plug = PlugDescriptor("outComp2")
	outComp3_ : OutComp3Plug = PlugDescriptor("outComp3")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "tripleShadingSwitch"
	apiTypeInt = 620
	apiTypeStr = "kTripleShadingSwitch"
	typeIdInt = 1398229043
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["defComp1", "defComp2", "defComp3", "default", "inComp1", "inComp2", "inComp3", "inShape", "inTriple", "input", "outComp1", "outComp2", "outComp3", "output"]
	nodeLeafPlugs = ["default", "input", "output"]
	pass

