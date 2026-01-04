

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
class BiasXPlug(Plug):
	parent : BiasPlug = PlugDescriptor("bias")
	node : Contrast = None
	pass
class BiasYPlug(Plug):
	parent : BiasPlug = PlugDescriptor("bias")
	node : Contrast = None
	pass
class BiasZPlug(Plug):
	parent : BiasPlug = PlugDescriptor("bias")
	node : Contrast = None
	pass
class BiasPlug(Plug):
	biasX_ : BiasXPlug = PlugDescriptor("biasX")
	bx_ : BiasXPlug = PlugDescriptor("biasX")
	biasY_ : BiasYPlug = PlugDescriptor("biasY")
	by_ : BiasYPlug = PlugDescriptor("biasY")
	biasZ_ : BiasZPlug = PlugDescriptor("biasZ")
	bz_ : BiasZPlug = PlugDescriptor("biasZ")
	node : Contrast = None
	pass
class BinMembershipPlug(Plug):
	node : Contrast = None
	pass
class ContrastXPlug(Plug):
	parent : ContrastPlug = PlugDescriptor("contrast")
	node : Contrast = None
	pass
class ContrastYPlug(Plug):
	parent : ContrastPlug = PlugDescriptor("contrast")
	node : Contrast = None
	pass
class ContrastZPlug(Plug):
	parent : ContrastPlug = PlugDescriptor("contrast")
	node : Contrast = None
	pass
class ContrastPlug(Plug):
	contrastX_ : ContrastXPlug = PlugDescriptor("contrastX")
	cx_ : ContrastXPlug = PlugDescriptor("contrastX")
	contrastY_ : ContrastYPlug = PlugDescriptor("contrastY")
	cy_ : ContrastYPlug = PlugDescriptor("contrastY")
	contrastZ_ : ContrastZPlug = PlugDescriptor("contrastZ")
	cz_ : ContrastZPlug = PlugDescriptor("contrastZ")
	node : Contrast = None
	pass
class OutValueXPlug(Plug):
	parent : OutValuePlug = PlugDescriptor("outValue")
	node : Contrast = None
	pass
class OutValueYPlug(Plug):
	parent : OutValuePlug = PlugDescriptor("outValue")
	node : Contrast = None
	pass
class OutValueZPlug(Plug):
	parent : OutValuePlug = PlugDescriptor("outValue")
	node : Contrast = None
	pass
class OutValuePlug(Plug):
	outValueX_ : OutValueXPlug = PlugDescriptor("outValueX")
	ox_ : OutValueXPlug = PlugDescriptor("outValueX")
	outValueY_ : OutValueYPlug = PlugDescriptor("outValueY")
	oy_ : OutValueYPlug = PlugDescriptor("outValueY")
	outValueZ_ : OutValueZPlug = PlugDescriptor("outValueZ")
	oz_ : OutValueZPlug = PlugDescriptor("outValueZ")
	node : Contrast = None
	pass
class RenderPassModePlug(Plug):
	node : Contrast = None
	pass
class ValueXPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Contrast = None
	pass
class ValueYPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Contrast = None
	pass
class ValueZPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Contrast = None
	pass
class ValuePlug(Plug):
	valueX_ : ValueXPlug = PlugDescriptor("valueX")
	vx_ : ValueXPlug = PlugDescriptor("valueX")
	valueY_ : ValueYPlug = PlugDescriptor("valueY")
	vy_ : ValueYPlug = PlugDescriptor("valueY")
	valueZ_ : ValueZPlug = PlugDescriptor("valueZ")
	vz_ : ValueZPlug = PlugDescriptor("valueZ")
	node : Contrast = None
	pass
# endregion


# define node class
class Contrast(_BASE_):
	biasX_ : BiasXPlug = PlugDescriptor("biasX")
	biasY_ : BiasYPlug = PlugDescriptor("biasY")
	biasZ_ : BiasZPlug = PlugDescriptor("biasZ")
	bias_ : BiasPlug = PlugDescriptor("bias")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	contrastX_ : ContrastXPlug = PlugDescriptor("contrastX")
	contrastY_ : ContrastYPlug = PlugDescriptor("contrastY")
	contrastZ_ : ContrastZPlug = PlugDescriptor("contrastZ")
	contrast_ : ContrastPlug = PlugDescriptor("contrast")
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

	typeName = "contrast"
	apiTypeInt = 38
	apiTypeStr = "kContrast"
	typeIdInt = 1380142926
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["biasX", "biasY", "biasZ", "bias", "binMembership", "contrastX", "contrastY", "contrastZ", "contrast", "outValueX", "outValueY", "outValueZ", "outValue", "renderPassMode", "valueX", "valueY", "valueZ", "value"]
	nodeLeafPlugs = ["bias", "binMembership", "contrast", "outValue", "renderPassMode", "value"]
	pass

