

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryFilter = retriever.getNodeCls("GeometryFilter")
assert GeometryFilter
if T.TYPE_CHECKING:
	from .. import GeometryFilter

# add node doc



# region plug type defs
class InPositionsPlug(Plug):
	node : HistorySwitch = None
	pass
class OutPositionsPlug(Plug):
	node : HistorySwitch = None
	pass
class PlayFromCachePlug(Plug):
	node : HistorySwitch = None
	pass
class UndeformedGeometryPlug(Plug):
	node : HistorySwitch = None
	pass
# endregion


# define node class
class HistorySwitch(GeometryFilter):
	inPositions_ : InPositionsPlug = PlugDescriptor("inPositions")
	outPositions_ : OutPositionsPlug = PlugDescriptor("outPositions")
	playFromCache_ : PlayFromCachePlug = PlugDescriptor("playFromCache")
	undeformedGeometry_ : UndeformedGeometryPlug = PlugDescriptor("undeformedGeometry")

	# node attributes

	typeName = "historySwitch"
	apiTypeInt = 988
	apiTypeStr = "kHistorySwitch"
	typeIdInt = 1212765011
	MFnCls = om.MFnGeometryFilter
	pass

