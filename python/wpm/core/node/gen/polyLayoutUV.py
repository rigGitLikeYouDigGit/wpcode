

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
class DenseLayoutPlug(Plug):
	node : PolyLayoutUV = None
	pass
class FlipReversedPlug(Plug):
	node : PolyLayoutUV = None
	pass
class GridUPlug(Plug):
	node : PolyLayoutUV = None
	pass
class GridVPlug(Plug):
	node : PolyLayoutUV = None
	pass
class LayoutPlug(Plug):
	node : PolyLayoutUV = None
	pass
class LayoutMethodPlug(Plug):
	node : PolyLayoutUV = None
	pass
class PercentageSpacePlug(Plug):
	node : PolyLayoutUV = None
	pass
class RotateForBestFitPlug(Plug):
	node : PolyLayoutUV = None
	pass
class ScalePlug(Plug):
	node : PolyLayoutUV = None
	pass
class SeparatePlug(Plug):
	node : PolyLayoutUV = None
	pass
class TwoSidedLayoutPlug(Plug):
	node : PolyLayoutUV = None
	pass
# endregion


# define node class
class PolyLayoutUV(PolyModifierUV):
	denseLayout_ : DenseLayoutPlug = PlugDescriptor("denseLayout")
	flipReversed_ : FlipReversedPlug = PlugDescriptor("flipReversed")
	gridU_ : GridUPlug = PlugDescriptor("gridU")
	gridV_ : GridVPlug = PlugDescriptor("gridV")
	layout_ : LayoutPlug = PlugDescriptor("layout")
	layoutMethod_ : LayoutMethodPlug = PlugDescriptor("layoutMethod")
	percentageSpace_ : PercentageSpacePlug = PlugDescriptor("percentageSpace")
	rotateForBestFit_ : RotateForBestFitPlug = PlugDescriptor("rotateForBestFit")
	scale_ : ScalePlug = PlugDescriptor("scale")
	separate_ : SeparatePlug = PlugDescriptor("separate")
	twoSidedLayout_ : TwoSidedLayoutPlug = PlugDescriptor("twoSidedLayout")

	# node attributes

	typeName = "polyLayoutUV"
	apiTypeInt = 852
	apiTypeStr = "kPolyLayoutUV"
	typeIdInt = 1347179862
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["denseLayout", "flipReversed", "gridU", "gridV", "layout", "layoutMethod", "percentageSpace", "rotateForBestFit", "scale", "separate", "twoSidedLayout"]
	nodeLeafPlugs = ["denseLayout", "flipReversed", "gridU", "gridV", "layout", "layoutMethod", "percentageSpace", "rotateForBestFit", "scale", "separate", "twoSidedLayout"]
	pass

