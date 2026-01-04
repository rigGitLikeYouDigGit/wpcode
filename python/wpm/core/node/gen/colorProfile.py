

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
	node : ColorProfile = None
	pass
class ColorProfileTypePlug(Plug):
	node : ColorProfile = None
	pass
class ColorTemperaturePlug(Plug):
	node : ColorProfile = None
	pass
class ForceGammaPlug(Plug):
	node : ColorProfile = None
	pass
class GammaPlug(Plug):
	node : ColorProfile = None
	pass
class GammaOffsetPlug(Plug):
	node : ColorProfile = None
	pass
class IntensityPlug(Plug):
	node : ColorProfile = None
	pass
class TransformRow1Col1Plug(Plug):
	parent : TransformRow1Plug = PlugDescriptor("transformRow1")
	node : ColorProfile = None
	pass
class TransformRow1Col2Plug(Plug):
	parent : TransformRow1Plug = PlugDescriptor("transformRow1")
	node : ColorProfile = None
	pass
class TransformRow1Col3Plug(Plug):
	parent : TransformRow1Plug = PlugDescriptor("transformRow1")
	node : ColorProfile = None
	pass
class TransformRow1Plug(Plug):
	parent : TransformPlug = PlugDescriptor("transform")
	transformRow1Col1_ : TransformRow1Col1Plug = PlugDescriptor("transformRow1Col1")
	tr11_ : TransformRow1Col1Plug = PlugDescriptor("transformRow1Col1")
	transformRow1Col2_ : TransformRow1Col2Plug = PlugDescriptor("transformRow1Col2")
	tr12_ : TransformRow1Col2Plug = PlugDescriptor("transformRow1Col2")
	transformRow1Col3_ : TransformRow1Col3Plug = PlugDescriptor("transformRow1Col3")
	tr13_ : TransformRow1Col3Plug = PlugDescriptor("transformRow1Col3")
	node : ColorProfile = None
	pass
class TransformRow2Col1Plug(Plug):
	parent : TransformRow2Plug = PlugDescriptor("transformRow2")
	node : ColorProfile = None
	pass
class TransformRow2Col2Plug(Plug):
	parent : TransformRow2Plug = PlugDescriptor("transformRow2")
	node : ColorProfile = None
	pass
class TransformRow2Col3Plug(Plug):
	parent : TransformRow2Plug = PlugDescriptor("transformRow2")
	node : ColorProfile = None
	pass
class TransformRow2Plug(Plug):
	parent : TransformPlug = PlugDescriptor("transform")
	transformRow2Col1_ : TransformRow2Col1Plug = PlugDescriptor("transformRow2Col1")
	tr21_ : TransformRow2Col1Plug = PlugDescriptor("transformRow2Col1")
	transformRow2Col2_ : TransformRow2Col2Plug = PlugDescriptor("transformRow2Col2")
	tr22_ : TransformRow2Col2Plug = PlugDescriptor("transformRow2Col2")
	transformRow2Col3_ : TransformRow2Col3Plug = PlugDescriptor("transformRow2Col3")
	tr23_ : TransformRow2Col3Plug = PlugDescriptor("transformRow2Col3")
	node : ColorProfile = None
	pass
class TransformRow3Col1Plug(Plug):
	parent : TransformRow3Plug = PlugDescriptor("transformRow3")
	node : ColorProfile = None
	pass
class TransformRow3Col2Plug(Plug):
	parent : TransformRow3Plug = PlugDescriptor("transformRow3")
	node : ColorProfile = None
	pass
class TransformRow3Col3Plug(Plug):
	parent : TransformRow3Plug = PlugDescriptor("transformRow3")
	node : ColorProfile = None
	pass
class TransformRow3Plug(Plug):
	parent : TransformPlug = PlugDescriptor("transform")
	transformRow3Col1_ : TransformRow3Col1Plug = PlugDescriptor("transformRow3Col1")
	tr31_ : TransformRow3Col1Plug = PlugDescriptor("transformRow3Col1")
	transformRow3Col2_ : TransformRow3Col2Plug = PlugDescriptor("transformRow3Col2")
	tr32_ : TransformRow3Col2Plug = PlugDescriptor("transformRow3Col2")
	transformRow3Col3_ : TransformRow3Col3Plug = PlugDescriptor("transformRow3Col3")
	tr33_ : TransformRow3Col3Plug = PlugDescriptor("transformRow3Col3")
	node : ColorProfile = None
	pass
class TransformPlug(Plug):
	transformRow1_ : TransformRow1Plug = PlugDescriptor("transformRow1")
	tr1_ : TransformRow1Plug = PlugDescriptor("transformRow1")
	transformRow2_ : TransformRow2Plug = PlugDescriptor("transformRow2")
	tr2_ : TransformRow2Plug = PlugDescriptor("transformRow2")
	transformRow3_ : TransformRow3Plug = PlugDescriptor("transformRow3")
	tr3_ : TransformRow3Plug = PlugDescriptor("transformRow3")
	node : ColorProfile = None
	pass
class WhitepointBPlug(Plug):
	parent : WhitepointPlug = PlugDescriptor("whitepoint")
	node : ColorProfile = None
	pass
class WhitepointGPlug(Plug):
	parent : WhitepointPlug = PlugDescriptor("whitepoint")
	node : ColorProfile = None
	pass
class WhitepointRPlug(Plug):
	parent : WhitepointPlug = PlugDescriptor("whitepoint")
	node : ColorProfile = None
	pass
class WhitepointPlug(Plug):
	whitepointB_ : WhitepointBPlug = PlugDescriptor("whitepointB")
	wpb_ : WhitepointBPlug = PlugDescriptor("whitepointB")
	whitepointG_ : WhitepointGPlug = PlugDescriptor("whitepointG")
	wpg_ : WhitepointGPlug = PlugDescriptor("whitepointG")
	whitepointR_ : WhitepointRPlug = PlugDescriptor("whitepointR")
	wpr_ : WhitepointRPlug = PlugDescriptor("whitepointR")
	node : ColorProfile = None
	pass
# endregion


# define node class
class ColorProfile(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	colorProfileType_ : ColorProfileTypePlug = PlugDescriptor("colorProfileType")
	colorTemperature_ : ColorTemperaturePlug = PlugDescriptor("colorTemperature")
	forceGamma_ : ForceGammaPlug = PlugDescriptor("forceGamma")
	gamma_ : GammaPlug = PlugDescriptor("gamma")
	gammaOffset_ : GammaOffsetPlug = PlugDescriptor("gammaOffset")
	intensity_ : IntensityPlug = PlugDescriptor("intensity")
	transformRow1Col1_ : TransformRow1Col1Plug = PlugDescriptor("transformRow1Col1")
	transformRow1Col2_ : TransformRow1Col2Plug = PlugDescriptor("transformRow1Col2")
	transformRow1Col3_ : TransformRow1Col3Plug = PlugDescriptor("transformRow1Col3")
	transformRow1_ : TransformRow1Plug = PlugDescriptor("transformRow1")
	transformRow2Col1_ : TransformRow2Col1Plug = PlugDescriptor("transformRow2Col1")
	transformRow2Col2_ : TransformRow2Col2Plug = PlugDescriptor("transformRow2Col2")
	transformRow2Col3_ : TransformRow2Col3Plug = PlugDescriptor("transformRow2Col3")
	transformRow2_ : TransformRow2Plug = PlugDescriptor("transformRow2")
	transformRow3Col1_ : TransformRow3Col1Plug = PlugDescriptor("transformRow3Col1")
	transformRow3Col2_ : TransformRow3Col2Plug = PlugDescriptor("transformRow3Col2")
	transformRow3Col3_ : TransformRow3Col3Plug = PlugDescriptor("transformRow3Col3")
	transformRow3_ : TransformRow3Plug = PlugDescriptor("transformRow3")
	transform_ : TransformPlug = PlugDescriptor("transform")
	whitepointB_ : WhitepointBPlug = PlugDescriptor("whitepointB")
	whitepointG_ : WhitepointGPlug = PlugDescriptor("whitepointG")
	whitepointR_ : WhitepointRPlug = PlugDescriptor("whitepointR")
	whitepoint_ : WhitepointPlug = PlugDescriptor("whitepoint")

	# node attributes

	typeName = "colorProfile"
	apiTypeInt = 1066
	apiTypeStr = "kColorProfile"
	typeIdInt = 1129270352
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "colorProfileType", "colorTemperature", "forceGamma", "gamma", "gammaOffset", "intensity", "transformRow1Col1", "transformRow1Col2", "transformRow1Col3", "transformRow1", "transformRow2Col1", "transformRow2Col2", "transformRow2Col3", "transformRow2", "transformRow3Col1", "transformRow3Col2", "transformRow3Col3", "transformRow3", "transform", "whitepointB", "whitepointG", "whitepointR", "whitepoint"]
	nodeLeafPlugs = ["binMembership", "colorProfileType", "colorTemperature", "forceGamma", "gamma", "gammaOffset", "intensity", "transform", "whitepoint"]
	pass

