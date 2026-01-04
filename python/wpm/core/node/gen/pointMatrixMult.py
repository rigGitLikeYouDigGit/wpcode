

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : PointMatrixMult = None
	pass
class InMatrixPlug(Plug):
	node : PointMatrixMult = None
	pass
class InPointXPlug(Plug):
	parent : InPointPlug = PlugDescriptor("inPoint")
	node : PointMatrixMult = None
	pass
class InPointYPlug(Plug):
	parent : InPointPlug = PlugDescriptor("inPoint")
	node : PointMatrixMult = None
	pass
class InPointZPlug(Plug):
	parent : InPointPlug = PlugDescriptor("inPoint")
	node : PointMatrixMult = None
	pass
class InPointPlug(Plug):
	inPointX_ : InPointXPlug = PlugDescriptor("inPointX")
	ipx_ : InPointXPlug = PlugDescriptor("inPointX")
	inPointY_ : InPointYPlug = PlugDescriptor("inPointY")
	ipy_ : InPointYPlug = PlugDescriptor("inPointY")
	inPointZ_ : InPointZPlug = PlugDescriptor("inPointZ")
	ipz_ : InPointZPlug = PlugDescriptor("inPointZ")
	node : PointMatrixMult = None
	pass
class OutputXPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : PointMatrixMult = None
	pass
class OutputYPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : PointMatrixMult = None
	pass
class OutputZPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : PointMatrixMult = None
	pass
class OutputPlug(Plug):
	outputX_ : OutputXPlug = PlugDescriptor("outputX")
	ox_ : OutputXPlug = PlugDescriptor("outputX")
	outputY_ : OutputYPlug = PlugDescriptor("outputY")
	oy_ : OutputYPlug = PlugDescriptor("outputY")
	outputZ_ : OutputZPlug = PlugDescriptor("outputZ")
	oz_ : OutputZPlug = PlugDescriptor("outputZ")
	node : PointMatrixMult = None
	pass
class VectorMultiplyPlug(Plug):
	node : PointMatrixMult = None
	pass
# endregion


# define node class
class PointMatrixMult(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inMatrix_ : InMatrixPlug = PlugDescriptor("inMatrix")
	inPointX_ : InPointXPlug = PlugDescriptor("inPointX")
	inPointY_ : InPointYPlug = PlugDescriptor("inPointY")
	inPointZ_ : InPointZPlug = PlugDescriptor("inPointZ")
	inPoint_ : InPointPlug = PlugDescriptor("inPoint")
	outputX_ : OutputXPlug = PlugDescriptor("outputX")
	outputY_ : OutputYPlug = PlugDescriptor("outputY")
	outputZ_ : OutputZPlug = PlugDescriptor("outputZ")
	output_ : OutputPlug = PlugDescriptor("output")
	vectorMultiply_ : VectorMultiplyPlug = PlugDescriptor("vectorMultiply")

	# node attributes

	typeName = "pointMatrixMult"
	apiTypeInt = 462
	apiTypeStr = "kPointMatrixMult"
	typeIdInt = 1146113357
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "inMatrix", "inPointX", "inPointY", "inPointZ", "inPoint", "outputX", "outputY", "outputZ", "output", "vectorMultiply"]
	nodeLeafPlugs = ["binMembership", "inMatrix", "inPoint", "output", "vectorMultiply"]
	pass

