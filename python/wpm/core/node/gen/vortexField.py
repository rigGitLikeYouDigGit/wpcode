

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Field = retriever.getNodeCls("Field")
assert Field
if T.TYPE_CHECKING:
	from .. import Field

# add node doc



# region plug type defs
class AxisXPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : VortexField = None
	pass
class AxisYPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : VortexField = None
	pass
class AxisZPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : VortexField = None
	pass
class AxisPlug(Plug):
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axx_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axy_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axz_ : AxisZPlug = PlugDescriptor("axisZ")
	node : VortexField = None
	pass
# endregion


# define node class
class VortexField(Field):
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axis_ : AxisPlug = PlugDescriptor("axis")

	# node attributes

	typeName = "vortexField"
	typeIdInt = 1498828626
	pass

