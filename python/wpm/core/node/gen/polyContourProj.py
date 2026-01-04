

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierUV = Catalogue.PolyModifierUV
else:
	from .. import retriever
	PolyModifierUV = retriever.getNodeCls("PolyModifierUV")
	assert PolyModifierUV

# add node doc



# region plug type defs
class CornerVerticesPlug(Plug):
	node : PolyContourProj = None
	pass
class FlipRailsPlug(Plug):
	node : PolyContourProj = None
	pass
class ManipPointsPlug(Plug):
	node : PolyContourProj = None
	pass
class MethodPlug(Plug):
	node : PolyContourProj = None
	pass
class Offset0Plug(Plug):
	node : PolyContourProj = None
	pass
class Offset1Plug(Plug):
	node : PolyContourProj = None
	pass
class Offset2Plug(Plug):
	node : PolyContourProj = None
	pass
class Offset3Plug(Plug):
	node : PolyContourProj = None
	pass
class ReduceShearPlug(Plug):
	node : PolyContourProj = None
	pass
class Smoothness0Plug(Plug):
	node : PolyContourProj = None
	pass
class Smoothness1Plug(Plug):
	node : PolyContourProj = None
	pass
class Smoothness2Plug(Plug):
	node : PolyContourProj = None
	pass
class Smoothness3Plug(Plug):
	node : PolyContourProj = None
	pass
class SurfacePlug(Plug):
	node : PolyContourProj = None
	pass
class UserDefinedCornersPlug(Plug):
	node : PolyContourProj = None
	pass
# endregion


# define node class
class PolyContourProj(PolyModifierUV):
	cornerVertices_ : CornerVerticesPlug = PlugDescriptor("cornerVertices")
	flipRails_ : FlipRailsPlug = PlugDescriptor("flipRails")
	manipPoints_ : ManipPointsPlug = PlugDescriptor("manipPoints")
	method_ : MethodPlug = PlugDescriptor("method")
	offset0_ : Offset0Plug = PlugDescriptor("offset0")
	offset1_ : Offset1Plug = PlugDescriptor("offset1")
	offset2_ : Offset2Plug = PlugDescriptor("offset2")
	offset3_ : Offset3Plug = PlugDescriptor("offset3")
	reduceShear_ : ReduceShearPlug = PlugDescriptor("reduceShear")
	smoothness0_ : Smoothness0Plug = PlugDescriptor("smoothness0")
	smoothness1_ : Smoothness1Plug = PlugDescriptor("smoothness1")
	smoothness2_ : Smoothness2Plug = PlugDescriptor("smoothness2")
	smoothness3_ : Smoothness3Plug = PlugDescriptor("smoothness3")
	surface_ : SurfacePlug = PlugDescriptor("surface")
	userDefinedCorners_ : UserDefinedCornersPlug = PlugDescriptor("userDefinedCorners")

	# node attributes

	typeName = "polyContourProj"
	apiTypeInt = 1114
	apiTypeStr = "kPolyContourProj"
	typeIdInt = 1346588240
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["cornerVertices", "flipRails", "manipPoints", "method", "offset0", "offset1", "offset2", "offset3", "reduceShear", "smoothness0", "smoothness1", "smoothness2", "smoothness3", "surface", "userDefinedCorners"]
	nodeLeafPlugs = ["cornerVertices", "flipRails", "manipPoints", "method", "offset0", "offset1", "offset2", "offset3", "reduceShear", "smoothness0", "smoothness1", "smoothness2", "smoothness3", "surface", "userDefinedCorners"]
	pass

