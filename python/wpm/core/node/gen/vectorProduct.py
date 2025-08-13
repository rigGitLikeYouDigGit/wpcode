

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
class Input1XPlug(Plug):
	parent : Input1Plug = PlugDescriptor("input1")
	node : VectorProduct = None
	pass
class Input1YPlug(Plug):
	parent : Input1Plug = PlugDescriptor("input1")
	node : VectorProduct = None
	pass
class Input1ZPlug(Plug):
	parent : Input1Plug = PlugDescriptor("input1")
	node : VectorProduct = None
	pass
class Input1Plug(Plug):
	input1X_ : Input1XPlug = PlugDescriptor("input1X")
	i1x_ : Input1XPlug = PlugDescriptor("input1X")
	input1Y_ : Input1YPlug = PlugDescriptor("input1Y")
	i1y_ : Input1YPlug = PlugDescriptor("input1Y")
	input1Z_ : Input1ZPlug = PlugDescriptor("input1Z")
	i1z_ : Input1ZPlug = PlugDescriptor("input1Z")
	node : VectorProduct = None
	pass
class Input2XPlug(Plug):
	parent : Input2Plug = PlugDescriptor("input2")
	node : VectorProduct = None
	pass
class Input2YPlug(Plug):
	parent : Input2Plug = PlugDescriptor("input2")
	node : VectorProduct = None
	pass
class Input2ZPlug(Plug):
	parent : Input2Plug = PlugDescriptor("input2")
	node : VectorProduct = None
	pass
class Input2Plug(Plug):
	input2X_ : Input2XPlug = PlugDescriptor("input2X")
	i2x_ : Input2XPlug = PlugDescriptor("input2X")
	input2Y_ : Input2YPlug = PlugDescriptor("input2Y")
	i2y_ : Input2YPlug = PlugDescriptor("input2Y")
	input2Z_ : Input2ZPlug = PlugDescriptor("input2Z")
	i2z_ : Input2ZPlug = PlugDescriptor("input2Z")
	node : VectorProduct = None
	pass
class MatrixPlug(Plug):
	node : VectorProduct = None
	pass
class NormalizeOutputPlug(Plug):
	node : VectorProduct = None
	pass
class OperationPlug(Plug):
	node : VectorProduct = None
	pass
class OutputXPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : VectorProduct = None
	pass
class OutputYPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : VectorProduct = None
	pass
class OutputZPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : VectorProduct = None
	pass
class OutputPlug(Plug):
	outputX_ : OutputXPlug = PlugDescriptor("outputX")
	ox_ : OutputXPlug = PlugDescriptor("outputX")
	outputY_ : OutputYPlug = PlugDescriptor("outputY")
	oy_ : OutputYPlug = PlugDescriptor("outputY")
	outputZ_ : OutputZPlug = PlugDescriptor("outputZ")
	oz_ : OutputZPlug = PlugDescriptor("outputZ")
	node : VectorProduct = None
	pass
# endregion


# define node class
class VectorProduct(ShadingDependNode):
	input1X_ : Input1XPlug = PlugDescriptor("input1X")
	input1Y_ : Input1YPlug = PlugDescriptor("input1Y")
	input1Z_ : Input1ZPlug = PlugDescriptor("input1Z")
	input1_ : Input1Plug = PlugDescriptor("input1")
	input2X_ : Input2XPlug = PlugDescriptor("input2X")
	input2Y_ : Input2YPlug = PlugDescriptor("input2Y")
	input2Z_ : Input2ZPlug = PlugDescriptor("input2Z")
	input2_ : Input2Plug = PlugDescriptor("input2")
	matrix_ : MatrixPlug = PlugDescriptor("matrix")
	normalizeOutput_ : NormalizeOutputPlug = PlugDescriptor("normalizeOutput")
	operation_ : OperationPlug = PlugDescriptor("operation")
	outputX_ : OutputXPlug = PlugDescriptor("outputX")
	outputY_ : OutputYPlug = PlugDescriptor("outputY")
	outputZ_ : OutputZPlug = PlugDescriptor("outputZ")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "vectorProduct"
	apiTypeInt = 533
	apiTypeStr = "kVectorProduct"
	typeIdInt = 1381385539
	MFnCls = om.MFnDependencyNode
	pass

