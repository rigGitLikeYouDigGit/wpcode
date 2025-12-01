

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ShadingDependNode = retriever.getNodeCls("ShadingDependNode")
assert ShadingDependNode
if T.TYPE_CHECKING:
	from .. import ShadingDependNode

# add node doc



# region plug type defs
class Input1DPlug(Plug):
	node : PlusMinusAverage = None
	pass
class Input2DxPlug(Plug):
	parent : Input2DPlug = PlugDescriptor("input2D")
	node : PlusMinusAverage = None
	pass
class Input2DyPlug(Plug):
	parent : Input2DPlug = PlugDescriptor("input2D")
	node : PlusMinusAverage = None
	pass
class Input2DPlug(Plug):
	input2Dx_ : Input2DxPlug = PlugDescriptor("input2Dx")
	i2x_ : Input2DxPlug = PlugDescriptor("input2Dx")
	input2Dy_ : Input2DyPlug = PlugDescriptor("input2Dy")
	i2y_ : Input2DyPlug = PlugDescriptor("input2Dy")
	node : PlusMinusAverage = None
	pass
class Input3DxPlug(Plug):
	parent : Input3DPlug = PlugDescriptor("input3D")
	node : PlusMinusAverage = None
	pass
class Input3DyPlug(Plug):
	parent : Input3DPlug = PlugDescriptor("input3D")
	node : PlusMinusAverage = None
	pass
class Input3DzPlug(Plug):
	parent : Input3DPlug = PlugDescriptor("input3D")
	node : PlusMinusAverage = None
	pass
class Input3DPlug(Plug):
	input3Dx_ : Input3DxPlug = PlugDescriptor("input3Dx")
	i3x_ : Input3DxPlug = PlugDescriptor("input3Dx")
	input3Dy_ : Input3DyPlug = PlugDescriptor("input3Dy")
	i3y_ : Input3DyPlug = PlugDescriptor("input3Dy")
	input3Dz_ : Input3DzPlug = PlugDescriptor("input3Dz")
	i3z_ : Input3DzPlug = PlugDescriptor("input3Dz")
	node : PlusMinusAverage = None
	pass
class OperationPlug(Plug):
	node : PlusMinusAverage = None
	pass
class Output1DPlug(Plug):
	node : PlusMinusAverage = None
	pass
class Output2DxPlug(Plug):
	parent : Output2DPlug = PlugDescriptor("output2D")
	node : PlusMinusAverage = None
	pass
class Output2DyPlug(Plug):
	parent : Output2DPlug = PlugDescriptor("output2D")
	node : PlusMinusAverage = None
	pass
class Output2DPlug(Plug):
	output2Dx_ : Output2DxPlug = PlugDescriptor("output2Dx")
	o2x_ : Output2DxPlug = PlugDescriptor("output2Dx")
	output2Dy_ : Output2DyPlug = PlugDescriptor("output2Dy")
	o2y_ : Output2DyPlug = PlugDescriptor("output2Dy")
	node : PlusMinusAverage = None
	pass
class Output3DxPlug(Plug):
	parent : Output3DPlug = PlugDescriptor("output3D")
	node : PlusMinusAverage = None
	pass
class Output3DyPlug(Plug):
	parent : Output3DPlug = PlugDescriptor("output3D")
	node : PlusMinusAverage = None
	pass
class Output3DzPlug(Plug):
	parent : Output3DPlug = PlugDescriptor("output3D")
	node : PlusMinusAverage = None
	pass
class Output3DPlug(Plug):
	output3Dx_ : Output3DxPlug = PlugDescriptor("output3Dx")
	o3x_ : Output3DxPlug = PlugDescriptor("output3Dx")
	output3Dy_ : Output3DyPlug = PlugDescriptor("output3Dy")
	o3y_ : Output3DyPlug = PlugDescriptor("output3Dy")
	output3Dz_ : Output3DzPlug = PlugDescriptor("output3Dz")
	o3z_ : Output3DzPlug = PlugDescriptor("output3Dz")
	node : PlusMinusAverage = None
	pass
# endregion


# define node class
class PlusMinusAverage(ShadingDependNode):
	input1D_ : Input1DPlug = PlugDescriptor("input1D")
	input2Dx_ : Input2DxPlug = PlugDescriptor("input2Dx")
	input2Dy_ : Input2DyPlug = PlugDescriptor("input2Dy")
	input2D_ : Input2DPlug = PlugDescriptor("input2D")
	input3Dx_ : Input3DxPlug = PlugDescriptor("input3Dx")
	input3Dy_ : Input3DyPlug = PlugDescriptor("input3Dy")
	input3Dz_ : Input3DzPlug = PlugDescriptor("input3Dz")
	input3D_ : Input3DPlug = PlugDescriptor("input3D")
	operation_ : OperationPlug = PlugDescriptor("operation")
	output1D_ : Output1DPlug = PlugDescriptor("output1D")
	output2Dx_ : Output2DxPlug = PlugDescriptor("output2Dx")
	output2Dy_ : Output2DyPlug = PlugDescriptor("output2Dy")
	output2D_ : Output2DPlug = PlugDescriptor("output2D")
	output3Dx_ : Output3DxPlug = PlugDescriptor("output3Dx")
	output3Dy_ : Output3DyPlug = PlugDescriptor("output3Dy")
	output3Dz_ : Output3DzPlug = PlugDescriptor("output3Dz")
	output3D_ : Output3DPlug = PlugDescriptor("output3D")

	# node attributes

	typeName = "plusMinusAverage"
	apiTypeInt = 461
	apiTypeStr = "kPlusMinusAverage"
	typeIdInt = 1380994369
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["input1D", "input2Dx", "input2Dy", "input2D", "input3Dx", "input3Dy", "input3Dz", "input3D", "operation", "output1D", "output2Dx", "output2Dy", "output2D", "output3Dx", "output3Dy", "output3Dz", "output3D"]
	nodeLeafPlugs = ["input1D", "input2D", "input3D", "operation", "output1D", "output2D", "output3D"]
	pass

