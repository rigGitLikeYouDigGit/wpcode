

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
class CenterXPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : MakeNurbsSquare = None
	pass
class CenterYPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : MakeNurbsSquare = None
	pass
class CenterZPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : MakeNurbsSquare = None
	pass
class CenterPlug(Plug):
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	cx_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	cy_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	cz_ : CenterZPlug = PlugDescriptor("centerZ")
	node : MakeNurbsSquare = None
	pass
class DegreePlug(Plug):
	node : MakeNurbsSquare = None
	pass
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : MakeNurbsSquare = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : MakeNurbsSquare = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : MakeNurbsSquare = None
	pass
class NormalPlug(Plug):
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nrx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	nry_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nrz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : MakeNurbsSquare = None
	pass
class OutputCurve1Plug(Plug):
	node : MakeNurbsSquare = None
	pass
class OutputCurve2Plug(Plug):
	node : MakeNurbsSquare = None
	pass
class OutputCurve3Plug(Plug):
	node : MakeNurbsSquare = None
	pass
class OutputCurve4Plug(Plug):
	node : MakeNurbsSquare = None
	pass
class SideLength1Plug(Plug):
	node : MakeNurbsSquare = None
	pass
class SideLength2Plug(Plug):
	node : MakeNurbsSquare = None
	pass
class SpansPerSidePlug(Plug):
	node : MakeNurbsSquare = None
	pass
# endregion


# define node class
class MakeNurbsSquare(AbstractBaseCreate):
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	center_ : CenterPlug = PlugDescriptor("center")
	degree_ : DegreePlug = PlugDescriptor("degree")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	outputCurve1_ : OutputCurve1Plug = PlugDescriptor("outputCurve1")
	outputCurve2_ : OutputCurve2Plug = PlugDescriptor("outputCurve2")
	outputCurve3_ : OutputCurve3Plug = PlugDescriptor("outputCurve3")
	outputCurve4_ : OutputCurve4Plug = PlugDescriptor("outputCurve4")
	sideLength1_ : SideLength1Plug = PlugDescriptor("sideLength1")
	sideLength2_ : SideLength2Plug = PlugDescriptor("sideLength2")
	spansPerSide_ : SpansPerSidePlug = PlugDescriptor("spansPerSide")

	# node attributes

	typeName = "makeNurbsSquare"
	typeIdInt = 1314083154
	nodeLeafClassAttrs = ["centerX", "centerY", "centerZ", "center", "degree", "normalX", "normalY", "normalZ", "normal", "outputCurve1", "outputCurve2", "outputCurve3", "outputCurve4", "sideLength1", "sideLength2", "spansPerSide"]
	nodeLeafPlugs = ["center", "degree", "normal", "outputCurve1", "outputCurve2", "outputCurve3", "outputCurve4", "sideLength1", "sideLength2", "spansPerSide"]
	pass

