

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class CleanEdgesPlug(Plug):
	node : PolyClean = None
	pass
class CleanPartialUVMappingPlug(Plug):
	node : PolyClean = None
	pass
class CleanUVsPlug(Plug):
	node : PolyClean = None
	pass
class CleanVerticesPlug(Plug):
	node : PolyClean = None
	pass
# endregion


# define node class
class PolyClean(PolyModifier):
	cleanEdges_ : CleanEdgesPlug = PlugDescriptor("cleanEdges")
	cleanPartialUVMapping_ : CleanPartialUVMappingPlug = PlugDescriptor("cleanPartialUVMapping")
	cleanUVs_ : CleanUVsPlug = PlugDescriptor("cleanUVs")
	cleanVertices_ : CleanVerticesPlug = PlugDescriptor("cleanVertices")

	# node attributes

	typeName = "polyClean"
	apiTypeInt = 1124
	apiTypeStr = "kPolyClean"
	typeIdInt = 1347175244
	MFnCls = om.MFnDependencyNode
	pass

