

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class CurrentDriverPlug(Plug):
	node : FlexorShape = None
	pass
class DriverPlug(Plug):
	node : FlexorShape = None
	pass
class FlexorNodesPlug(Plug):
	node : FlexorShape = None
	pass
# endregion


# define node class
class FlexorShape(Shape):
	currentDriver_ : CurrentDriverPlug = PlugDescriptor("currentDriver")
	driver_ : DriverPlug = PlugDescriptor("driver")
	flexorNodes_ : FlexorNodesPlug = PlugDescriptor("flexorNodes")

	# node attributes

	typeName = "flexorShape"
	typeIdInt = 1179408456
	pass

