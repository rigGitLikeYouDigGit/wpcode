

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ParentTessellate = Catalogue.ParentTessellate
else:
	from .. import retriever
	ParentTessellate = retriever.getNodeCls("ParentTessellate")
	assert ParentTessellate

# add node doc



# region plug type defs
class CurvatureTolerancePlug(Plug):
	node : NurbsTessellate = None
	pass
class ExplicitTessellationAttributesPlug(Plug):
	node : NurbsTessellate = None
	pass
class InputSurfacePlug(Plug):
	node : NurbsTessellate = None
	pass
class SmoothEdgePlug(Plug):
	node : NurbsTessellate = None
	pass
class SmoothEdgeRatioPlug(Plug):
	node : NurbsTessellate = None
	pass
class UDivisionsFactorPlug(Plug):
	node : NurbsTessellate = None
	pass
class VDivisionsFactorPlug(Plug):
	node : NurbsTessellate = None
	pass
# endregion


# define node class
class NurbsTessellate(ParentTessellate):
	curvatureTolerance_ : CurvatureTolerancePlug = PlugDescriptor("curvatureTolerance")
	explicitTessellationAttributes_ : ExplicitTessellationAttributesPlug = PlugDescriptor("explicitTessellationAttributes")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	smoothEdge_ : SmoothEdgePlug = PlugDescriptor("smoothEdge")
	smoothEdgeRatio_ : SmoothEdgeRatioPlug = PlugDescriptor("smoothEdgeRatio")
	uDivisionsFactor_ : UDivisionsFactorPlug = PlugDescriptor("uDivisionsFactor")
	vDivisionsFactor_ : VDivisionsFactorPlug = PlugDescriptor("vDivisionsFactor")

	# node attributes

	typeName = "nurbsTessellate"
	typeIdInt = 1314145619
	nodeLeafClassAttrs = ["curvatureTolerance", "explicitTessellationAttributes", "inputSurface", "smoothEdge", "smoothEdgeRatio", "uDivisionsFactor", "vDivisionsFactor"]
	nodeLeafPlugs = ["curvatureTolerance", "explicitTessellationAttributes", "inputSurface", "smoothEdge", "smoothEdgeRatio", "uDivisionsFactor", "vDivisionsFactor"]
	pass

