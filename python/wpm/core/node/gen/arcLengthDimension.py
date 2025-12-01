

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
NurbsDimShape = retriever.getNodeCls("NurbsDimShape")
assert NurbsDimShape
if T.TYPE_CHECKING:
	from .. import NurbsDimShape

# add node doc



# region plug type defs
class ArcLengthPlug(Plug):
	node : ArcLengthDimension = None
	pass
class ArcLengthInVPlug(Plug):
	node : ArcLengthDimension = None
	pass
# endregion


# define node class
class ArcLengthDimension(NurbsDimShape):
	arcLength_ : ArcLengthPlug = PlugDescriptor("arcLength")
	arcLengthInV_ : ArcLengthInVPlug = PlugDescriptor("arcLengthInV")

	# node attributes

	typeName = "arcLengthDimension"
	typeIdInt = 1094995278
	nodeLeafClassAttrs = ["arcLength", "arcLengthInV"]
	nodeLeafPlugs = ["arcLength", "arcLengthInV"]
	pass

