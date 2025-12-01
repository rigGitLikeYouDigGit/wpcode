

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
	parent : DefTriplePlug = PlugDescriptor("defTriple")
	node : QuadShadingSwitch = None
	pass
class DefComp2Plug(Plug):
	parent : DefTriplePlug = PlugDescriptor("defTriple")
	node : QuadShadingSwitch = None
	pass
class DefComp3Plug(Plug):
	parent : DefTriplePlug = PlugDescriptor("defTriple")
	node : QuadShadingSwitch = None
	pass
class DefSinglePlug(Plug):
	parent : DefaultPlug = PlugDescriptor("default")
	node : QuadShadingSwitch = None
	pass
class DefTriplePlug(Plug):
	parent : DefaultPlug = PlugDescriptor("default")
	defComp1_ : DefComp1Plug = PlugDescriptor("defComp1")
	dc1_ : DefComp1Plug = PlugDescriptor("defComp1")
	defComp2_ : DefComp2Plug = PlugDescriptor("defComp2")
	dc2_ : DefComp2Plug = PlugDescriptor("defComp2")
	defComp3_ : DefComp3Plug = PlugDescriptor("defComp3")
	dc3_ : DefComp3Plug = PlugDescriptor("defComp3")
	node : QuadShadingSwitch = None
	pass
class DefaultPlug(Plug):
	defSingle_ : DefSinglePlug = PlugDescriptor("defSingle")
	dsi_ : DefSinglePlug = PlugDescriptor("defSingle")
	defTriple_ : DefTriplePlug = PlugDescriptor("defTriple")
	dtr_ : DefTriplePlug = PlugDescriptor("defTriple")
	node : QuadShadingSwitch = None
	pass
class InComp1Plug(Plug):
	parent : InTriplePlug = PlugDescriptor("inTriple")
	node : QuadShadingSwitch = None
	pass
class InComp2Plug(Plug):
	parent : InTriplePlug = PlugDescriptor("inTriple")
	node : QuadShadingSwitch = None
	pass
class InComp3Plug(Plug):
	parent : InTriplePlug = PlugDescriptor("inTriple")
	node : QuadShadingSwitch = None
	pass
class InSinglePlug(Plug):
	parent : InQuadPlug = PlugDescriptor("inQuad")
	node : QuadShadingSwitch = None
	pass
class InTriplePlug(Plug):
	parent : InQuadPlug = PlugDescriptor("inQuad")
	inComp1_ : InComp1Plug = PlugDescriptor("inComp1")
	ic1_ : InComp1Plug = PlugDescriptor("inComp1")
	inComp2_ : InComp2Plug = PlugDescriptor("inComp2")
	ic2_ : InComp2Plug = PlugDescriptor("inComp2")
	inComp3_ : InComp3Plug = PlugDescriptor("inComp3")
	ic3_ : InComp3Plug = PlugDescriptor("inComp3")
	node : QuadShadingSwitch = None
	pass
class InQuadPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	inSingle_ : InSinglePlug = PlugDescriptor("inSingle")
	isi_ : InSinglePlug = PlugDescriptor("inSingle")
	inTriple_ : InTriplePlug = PlugDescriptor("inTriple")
	itr_ : InTriplePlug = PlugDescriptor("inTriple")
	node : QuadShadingSwitch = None
	pass
class InShapePlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : QuadShadingSwitch = None
	pass
class InputPlug(Plug):
	inQuad_ : InQuadPlug = PlugDescriptor("inQuad")
	iq_ : InQuadPlug = PlugDescriptor("inQuad")
	inShape_ : InShapePlug = PlugDescriptor("inShape")
	is_ : InShapePlug = PlugDescriptor("inShape")
	node : QuadShadingSwitch = None
	pass
class OutComp1Plug(Plug):
	parent : OutTriplePlug = PlugDescriptor("outTriple")
	node : QuadShadingSwitch = None
	pass
class OutComp2Plug(Plug):
	parent : OutTriplePlug = PlugDescriptor("outTriple")
	node : QuadShadingSwitch = None
	pass
class OutComp3Plug(Plug):
	parent : OutTriplePlug = PlugDescriptor("outTriple")
	node : QuadShadingSwitch = None
	pass
class OutSinglePlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : QuadShadingSwitch = None
	pass
class OutTriplePlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	outComp1_ : OutComp1Plug = PlugDescriptor("outComp1")
	oc1_ : OutComp1Plug = PlugDescriptor("outComp1")
	outComp2_ : OutComp2Plug = PlugDescriptor("outComp2")
	oc2_ : OutComp2Plug = PlugDescriptor("outComp2")
	outComp3_ : OutComp3Plug = PlugDescriptor("outComp3")
	oc3_ : OutComp3Plug = PlugDescriptor("outComp3")
	node : QuadShadingSwitch = None
	pass
class OutputPlug(Plug):
	outSingle_ : OutSinglePlug = PlugDescriptor("outSingle")
	osi_ : OutSinglePlug = PlugDescriptor("outSingle")
	outTriple_ : OutTriplePlug = PlugDescriptor("outTriple")
	otr_ : OutTriplePlug = PlugDescriptor("outTriple")
	node : QuadShadingSwitch = None
	pass
# endregion


# define node class
class QuadShadingSwitch(BaseShadingSwitch):
	defComp1_ : DefComp1Plug = PlugDescriptor("defComp1")
	defComp2_ : DefComp2Plug = PlugDescriptor("defComp2")
	defComp3_ : DefComp3Plug = PlugDescriptor("defComp3")
	defSingle_ : DefSinglePlug = PlugDescriptor("defSingle")
	defTriple_ : DefTriplePlug = PlugDescriptor("defTriple")
	default_ : DefaultPlug = PlugDescriptor("default")
	inComp1_ : InComp1Plug = PlugDescriptor("inComp1")
	inComp2_ : InComp2Plug = PlugDescriptor("inComp2")
	inComp3_ : InComp3Plug = PlugDescriptor("inComp3")
	inSingle_ : InSinglePlug = PlugDescriptor("inSingle")
	inTriple_ : InTriplePlug = PlugDescriptor("inTriple")
	inQuad_ : InQuadPlug = PlugDescriptor("inQuad")
	inShape_ : InShapePlug = PlugDescriptor("inShape")
	input_ : InputPlug = PlugDescriptor("input")
	outComp1_ : OutComp1Plug = PlugDescriptor("outComp1")
	outComp2_ : OutComp2Plug = PlugDescriptor("outComp2")
	outComp3_ : OutComp3Plug = PlugDescriptor("outComp3")
	outSingle_ : OutSinglePlug = PlugDescriptor("outSingle")
	outTriple_ : OutTriplePlug = PlugDescriptor("outTriple")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "quadShadingSwitch"
	apiTypeInt = 925
	apiTypeStr = "kQuadShadingSwitch"
	typeIdInt = 1398229044
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["defComp1", "defComp2", "defComp3", "defSingle", "defTriple", "default", "inComp1", "inComp2", "inComp3", "inSingle", "inTriple", "inQuad", "inShape", "input", "outComp1", "outComp2", "outComp3", "outSingle", "outTriple", "output"]
	nodeLeafPlugs = ["default", "input", "output"]
	pass

