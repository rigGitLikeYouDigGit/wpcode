

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class AxisXPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : Primitive = None
	pass
class AxisYPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : Primitive = None
	pass
class AxisZPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : Primitive = None
	pass
class AxisPlug(Plug):
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axx_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axy_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axz_ : AxisZPlug = PlugDescriptor("axisZ")
	node : Primitive = None
	pass
class OutputSurfacePlug(Plug):
	node : Primitive = None
	pass
class PivotXPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : Primitive = None
	pass
class PivotYPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : Primitive = None
	pass
class PivotZPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : Primitive = None
	pass
class PivotPlug(Plug):
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	px_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	py_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pz_ : PivotZPlug = PlugDescriptor("pivotZ")
	node : Primitive = None
	pass
# endregion


# define node class
class Primitive(AbstractBaseCreate):
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axis_ : AxisPlug = PlugDescriptor("axis")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pivot_ : PivotPlug = PlugDescriptor("pivot")

	# node attributes

	typeName = "primitive"
	typeIdInt = 1313886797
	nodeLeafClassAttrs = ["axisX", "axisY", "axisZ", "axis", "outputSurface", "pivotX", "pivotY", "pivotZ", "pivot"]
	nodeLeafPlugs = ["axis", "outputSurface", "pivot"]
	pass

