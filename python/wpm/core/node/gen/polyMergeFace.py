

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifier = Catalogue.PolyModifier
else:
	from .. import retriever
	PolyModifier = retriever.getNodeCls("PolyModifier")
	assert PolyModifier

# add node doc



# region plug type defs
class FirstFacetPlug(Plug):
	node : PolyMergeFace = None
	pass
class MergeModePlug(Plug):
	node : PolyMergeFace = None
	pass
class SecondFacetPlug(Plug):
	node : PolyMergeFace = None
	pass
class UseAreaTolerancePlug(Plug):
	node : PolyMergeFace = None
	pass
# endregion


# define node class
class PolyMergeFace(PolyModifier):
	firstFacet_ : FirstFacetPlug = PlugDescriptor("firstFacet")
	mergeMode_ : MergeModePlug = PlugDescriptor("mergeMode")
	secondFacet_ : SecondFacetPlug = PlugDescriptor("secondFacet")
	useAreaTolerance_ : UseAreaTolerancePlug = PlugDescriptor("useAreaTolerance")

	# node attributes

	typeName = "polyMergeFace"
	typeIdInt = 1347241286
	nodeLeafClassAttrs = ["firstFacet", "mergeMode", "secondFacet", "useAreaTolerance"]
	nodeLeafPlugs = ["firstFacet", "mergeMode", "secondFacet", "useAreaTolerance"]
	pass

