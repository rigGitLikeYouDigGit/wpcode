

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
	node : GammaCorrect = None
	pass
class GammaXPlug(Plug):
	parent : GammaPlug = PlugDescriptor("gamma")
	node : GammaCorrect = None
	pass
class GammaYPlug(Plug):
	parent : GammaPlug = PlugDescriptor("gamma")
	node : GammaCorrect = None
	pass
class GammaZPlug(Plug):
	parent : GammaPlug = PlugDescriptor("gamma")
	node : GammaCorrect = None
	pass
class GammaPlug(Plug):
	gammaX_ : GammaXPlug = PlugDescriptor("gammaX")
	gx_ : GammaXPlug = PlugDescriptor("gammaX")
	gammaY_ : GammaYPlug = PlugDescriptor("gammaY")
	gy_ : GammaYPlug = PlugDescriptor("gammaY")
	gammaZ_ : GammaZPlug = PlugDescriptor("gammaZ")
	gz_ : GammaZPlug = PlugDescriptor("gammaZ")
	node : GammaCorrect = None
	pass
class OutValueXPlug(Plug):
	parent : OutValuePlug = PlugDescriptor("outValue")
	node : GammaCorrect = None
	pass
class OutValueYPlug(Plug):
	parent : OutValuePlug = PlugDescriptor("outValue")
	node : GammaCorrect = None
	pass
class OutValueZPlug(Plug):
	parent : OutValuePlug = PlugDescriptor("outValue")
	node : GammaCorrect = None
	pass
class OutValuePlug(Plug):
	outValueX_ : OutValueXPlug = PlugDescriptor("outValueX")
	ox_ : OutValueXPlug = PlugDescriptor("outValueX")
	outValueY_ : OutValueYPlug = PlugDescriptor("outValueY")
	oy_ : OutValueYPlug = PlugDescriptor("outValueY")
	outValueZ_ : OutValueZPlug = PlugDescriptor("outValueZ")
	oz_ : OutValueZPlug = PlugDescriptor("outValueZ")
	node : GammaCorrect = None
	pass
class RenderPassModePlug(Plug):
	node : GammaCorrect = None
	pass
class ValueXPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : GammaCorrect = None
	pass
class ValueYPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : GammaCorrect = None
	pass
class ValueZPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : GammaCorrect = None
	pass
class ValuePlug(Plug):
	valueX_ : ValueXPlug = PlugDescriptor("valueX")
	vx_ : ValueXPlug = PlugDescriptor("valueX")
	valueY_ : ValueYPlug = PlugDescriptor("valueY")
	vy_ : ValueYPlug = PlugDescriptor("valueY")
	valueZ_ : ValueZPlug = PlugDescriptor("valueZ")
	vz_ : ValueZPlug = PlugDescriptor("valueZ")
	node : GammaCorrect = None
	pass
# endregion


# define node class
class GammaCorrect(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	gammaX_ : GammaXPlug = PlugDescriptor("gammaX")
	gammaY_ : GammaYPlug = PlugDescriptor("gammaY")
	gammaZ_ : GammaZPlug = PlugDescriptor("gammaZ")
	gamma_ : GammaPlug = PlugDescriptor("gamma")
	outValueX_ : OutValueXPlug = PlugDescriptor("outValueX")
	outValueY_ : OutValueYPlug = PlugDescriptor("outValueY")
	outValueZ_ : OutValueZPlug = PlugDescriptor("outValueZ")
	outValue_ : OutValuePlug = PlugDescriptor("outValue")
	renderPassMode_ : RenderPassModePlug = PlugDescriptor("renderPassMode")
	valueX_ : ValueXPlug = PlugDescriptor("valueX")
	valueY_ : ValueYPlug = PlugDescriptor("valueY")
	valueZ_ : ValueZPlug = PlugDescriptor("valueZ")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "gammaCorrect"
	apiTypeInt = 333
	apiTypeStr = "kGammaCorrect"
	typeIdInt = 1380401485
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "gammaX", "gammaY", "gammaZ", "gamma", "outValueX", "outValueY", "outValueZ", "outValue", "renderPassMode", "valueX", "valueY", "valueZ", "value"]
	nodeLeafPlugs = ["binMembership", "gamma", "outValue", "renderPassMode", "value"]
	pass

