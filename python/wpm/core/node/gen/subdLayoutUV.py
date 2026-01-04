

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	SubdModifierUV = Catalogue.SubdModifierUV
else:
	from .. import retriever
	SubdModifierUV = retriever.getNodeCls("SubdModifierUV")
	assert SubdModifierUV

# add node doc



# region plug type defs
class DenseLayoutPlug(Plug):
	node : SubdLayoutUV = None
	pass
class FlipReversedPlug(Plug):
	node : SubdLayoutUV = None
	pass
class LayoutPlug(Plug):
	node : SubdLayoutUV = None
	pass
class LayoutMethodPlug(Plug):
	node : SubdLayoutUV = None
	pass
class PercentageSpacePlug(Plug):
	node : SubdLayoutUV = None
	pass
class RotateForBestFitPlug(Plug):
	node : SubdLayoutUV = None
	pass
class ScalePlug(Plug):
	node : SubdLayoutUV = None
	pass
class SeparatePlug(Plug):
	node : SubdLayoutUV = None
	pass
# endregion


# define node class
class SubdLayoutUV(SubdModifierUV):
	denseLayout_ : DenseLayoutPlug = PlugDescriptor("denseLayout")
	flipReversed_ : FlipReversedPlug = PlugDescriptor("flipReversed")
	layout_ : LayoutPlug = PlugDescriptor("layout")
	layoutMethod_ : LayoutMethodPlug = PlugDescriptor("layoutMethod")
	percentageSpace_ : PercentageSpacePlug = PlugDescriptor("percentageSpace")
	rotateForBestFit_ : RotateForBestFitPlug = PlugDescriptor("rotateForBestFit")
	scale_ : ScalePlug = PlugDescriptor("scale")
	separate_ : SeparatePlug = PlugDescriptor("separate")

	# node attributes

	typeName = "subdLayoutUV"
	apiTypeInt = 873
	apiTypeStr = "kSubdLayoutUV"
	typeIdInt = 1397511510
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["denseLayout", "flipReversed", "layout", "layoutMethod", "percentageSpace", "rotateForBestFit", "scale", "separate"]
	nodeLeafPlugs = ["denseLayout", "flipReversed", "layout", "layoutMethod", "percentageSpace", "rotateForBestFit", "scale", "separate"]
	pass

