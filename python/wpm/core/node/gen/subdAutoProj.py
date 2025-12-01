

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SubdModifierUV = retriever.getNodeCls("SubdModifierUV")
assert SubdModifierUV
if T.TYPE_CHECKING:
	from .. import SubdModifierUV

# add node doc



# region plug type defs
class DenseLayoutPlug(Plug):
	node : SubdAutoProj = None
	pass
class LayoutPlug(Plug):
	node : SubdAutoProj = None
	pass
class LayoutMethodPlug(Plug):
	node : SubdAutoProj = None
	pass
class OptimizePlug(Plug):
	node : SubdAutoProj = None
	pass
class PercentageSpacePlug(Plug):
	node : SubdAutoProj = None
	pass
class PlanesPlug(Plug):
	node : SubdAutoProj = None
	pass
class ScalePlug(Plug):
	node : SubdAutoProj = None
	pass
class SkipIntersectPlug(Plug):
	node : SubdAutoProj = None
	pass
# endregion


# define node class
class SubdAutoProj(SubdModifierUV):
	denseLayout_ : DenseLayoutPlug = PlugDescriptor("denseLayout")
	layout_ : LayoutPlug = PlugDescriptor("layout")
	layoutMethod_ : LayoutMethodPlug = PlugDescriptor("layoutMethod")
	optimize_ : OptimizePlug = PlugDescriptor("optimize")
	percentageSpace_ : PercentageSpacePlug = PlugDescriptor("percentageSpace")
	planes_ : PlanesPlug = PlugDescriptor("planes")
	scale_ : ScalePlug = PlugDescriptor("scale")
	skipIntersect_ : SkipIntersectPlug = PlugDescriptor("skipIntersect")

	# node attributes

	typeName = "subdAutoProj"
	apiTypeInt = 877
	apiTypeStr = "kSubdAutoProj"
	typeIdInt = 1396790608
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["denseLayout", "layout", "layoutMethod", "optimize", "percentageSpace", "planes", "scale", "skipIntersect"]
	nodeLeafPlugs = ["denseLayout", "layout", "layoutMethod", "optimize", "percentageSpace", "planes", "scale", "skipIntersect"]
	pass

