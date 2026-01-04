

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Texture2d = Catalogue.Texture2d
else:
	from .. import retriever
	Texture2d = retriever.getNodeCls("Texture2d")
	assert Texture2d

# add node doc



# region plug type defs
class ColorKeyBPlug(Plug):
	parent : ColorKeyPlug = PlugDescriptor("colorKey")
	node : Stencil = None
	pass
class ColorKeyGPlug(Plug):
	parent : ColorKeyPlug = PlugDescriptor("colorKey")
	node : Stencil = None
	pass
class ColorKeyRPlug(Plug):
	parent : ColorKeyPlug = PlugDescriptor("colorKey")
	node : Stencil = None
	pass
class ColorKeyPlug(Plug):
	colorKeyB_ : ColorKeyBPlug = PlugDescriptor("colorKeyB")
	ckb_ : ColorKeyBPlug = PlugDescriptor("colorKeyB")
	colorKeyG_ : ColorKeyGPlug = PlugDescriptor("colorKeyG")
	ckg_ : ColorKeyGPlug = PlugDescriptor("colorKeyG")
	colorKeyR_ : ColorKeyRPlug = PlugDescriptor("colorKeyR")
	ckr_ : ColorKeyRPlug = PlugDescriptor("colorKeyR")
	node : Stencil = None
	pass
class EdgeBlendPlug(Plug):
	node : Stencil = None
	pass
class HueRangePlug(Plug):
	node : Stencil = None
	pass
class ImageBPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : Stencil = None
	pass
class ImageGPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : Stencil = None
	pass
class ImageRPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : Stencil = None
	pass
class ImagePlug(Plug):
	imageB_ : ImageBPlug = PlugDescriptor("imageB")
	imb_ : ImageBPlug = PlugDescriptor("imageB")
	imageG_ : ImageGPlug = PlugDescriptor("imageG")
	img_ : ImageGPlug = PlugDescriptor("imageG")
	imageR_ : ImageRPlug = PlugDescriptor("imageR")
	imr_ : ImageRPlug = PlugDescriptor("imageR")
	node : Stencil = None
	pass
class KeyMaskingPlug(Plug):
	node : Stencil = None
	pass
class MaskPlug(Plug):
	node : Stencil = None
	pass
class PositiveKeyPlug(Plug):
	node : Stencil = None
	pass
class SaturationRangePlug(Plug):
	node : Stencil = None
	pass
class ThresholdPlug(Plug):
	node : Stencil = None
	pass
class ValueRangePlug(Plug):
	node : Stencil = None
	pass
# endregion


# define node class
class Stencil(Texture2d):
	colorKeyB_ : ColorKeyBPlug = PlugDescriptor("colorKeyB")
	colorKeyG_ : ColorKeyGPlug = PlugDescriptor("colorKeyG")
	colorKeyR_ : ColorKeyRPlug = PlugDescriptor("colorKeyR")
	colorKey_ : ColorKeyPlug = PlugDescriptor("colorKey")
	edgeBlend_ : EdgeBlendPlug = PlugDescriptor("edgeBlend")
	hueRange_ : HueRangePlug = PlugDescriptor("hueRange")
	imageB_ : ImageBPlug = PlugDescriptor("imageB")
	imageG_ : ImageGPlug = PlugDescriptor("imageG")
	imageR_ : ImageRPlug = PlugDescriptor("imageR")
	image_ : ImagePlug = PlugDescriptor("image")
	keyMasking_ : KeyMaskingPlug = PlugDescriptor("keyMasking")
	mask_ : MaskPlug = PlugDescriptor("mask")
	positiveKey_ : PositiveKeyPlug = PlugDescriptor("positiveKey")
	saturationRange_ : SaturationRangePlug = PlugDescriptor("saturationRange")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")
	valueRange_ : ValueRangePlug = PlugDescriptor("valueRange")

	# node attributes

	typeName = "stencil"
	apiTypeInt = 505
	apiTypeStr = "kStencil"
	typeIdInt = 1381258068
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["colorKeyB", "colorKeyG", "colorKeyR", "colorKey", "edgeBlend", "hueRange", "imageB", "imageG", "imageR", "image", "keyMasking", "mask", "positiveKey", "saturationRange", "threshold", "valueRange"]
	nodeLeafPlugs = ["colorKey", "edgeBlend", "hueRange", "image", "keyMasking", "mask", "positiveKey", "saturationRange", "threshold", "valueRange"]
	pass

