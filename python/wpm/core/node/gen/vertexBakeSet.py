

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	BakeSet = Catalogue.BakeSet
else:
	from .. import retriever
	BakeSet = retriever.getNodeCls("BakeSet")
	assert BakeSet

# add node doc



# region plug type defs
class AlphaBlendingPlug(Plug):
	node : VertexBakeSet = None
	pass
class BakeColorPlug(Plug):
	node : VertexBakeSet = None
	pass
class ClampMaxPlug(Plug):
	node : VertexBakeSet = None
	pass
class ClampMinPlug(Plug):
	node : VertexBakeSet = None
	pass
class ColorBlendingPlug(Plug):
	node : VertexBakeSet = None
	pass
class ColorSetNamePlug(Plug):
	node : VertexBakeSet = None
	pass
class MaxAlphaPlug(Plug):
	node : VertexBakeSet = None
	pass
class MaxColorBPlug(Plug):
	parent : MaxColorPlug = PlugDescriptor("maxColor")
	node : VertexBakeSet = None
	pass
class MaxColorGPlug(Plug):
	parent : MaxColorPlug = PlugDescriptor("maxColor")
	node : VertexBakeSet = None
	pass
class MaxColorRPlug(Plug):
	parent : MaxColorPlug = PlugDescriptor("maxColor")
	node : VertexBakeSet = None
	pass
class MaxColorPlug(Plug):
	maxColorB_ : MaxColorBPlug = PlugDescriptor("maxColorB")
	hb_ : MaxColorBPlug = PlugDescriptor("maxColorB")
	maxColorG_ : MaxColorGPlug = PlugDescriptor("maxColorG")
	hg_ : MaxColorGPlug = PlugDescriptor("maxColorG")
	maxColorR_ : MaxColorRPlug = PlugDescriptor("maxColorR")
	hr_ : MaxColorRPlug = PlugDescriptor("maxColorR")
	node : VertexBakeSet = None
	pass
class MinAlphaPlug(Plug):
	node : VertexBakeSet = None
	pass
class MinColorBPlug(Plug):
	parent : MinColorPlug = PlugDescriptor("minColor")
	node : VertexBakeSet = None
	pass
class MinColorGPlug(Plug):
	parent : MinColorPlug = PlugDescriptor("minColor")
	node : VertexBakeSet = None
	pass
class MinColorRPlug(Plug):
	parent : MinColorPlug = PlugDescriptor("minColor")
	node : VertexBakeSet = None
	pass
class MinColorPlug(Plug):
	minColorB_ : MinColorBPlug = PlugDescriptor("minColorB")
	lb_ : MinColorBPlug = PlugDescriptor("minColorB")
	minColorG_ : MinColorGPlug = PlugDescriptor("minColorG")
	lg_ : MinColorGPlug = PlugDescriptor("minColorG")
	minColorR_ : MinColorRPlug = PlugDescriptor("minColorR")
	lr_ : MinColorRPlug = PlugDescriptor("minColorR")
	node : VertexBakeSet = None
	pass
class ScaleRgbaPlug(Plug):
	node : VertexBakeSet = None
	pass
class SharedVerticesPlug(Plug):
	node : VertexBakeSet = None
	pass
class UseFaceNormalsPlug(Plug):
	node : VertexBakeSet = None
	pass
# endregion


# define node class
class VertexBakeSet(BakeSet):
	alphaBlending_ : AlphaBlendingPlug = PlugDescriptor("alphaBlending")
	bakeColor_ : BakeColorPlug = PlugDescriptor("bakeColor")
	clampMax_ : ClampMaxPlug = PlugDescriptor("clampMax")
	clampMin_ : ClampMinPlug = PlugDescriptor("clampMin")
	colorBlending_ : ColorBlendingPlug = PlugDescriptor("colorBlending")
	colorSetName_ : ColorSetNamePlug = PlugDescriptor("colorSetName")
	maxAlpha_ : MaxAlphaPlug = PlugDescriptor("maxAlpha")
	maxColorB_ : MaxColorBPlug = PlugDescriptor("maxColorB")
	maxColorG_ : MaxColorGPlug = PlugDescriptor("maxColorG")
	maxColorR_ : MaxColorRPlug = PlugDescriptor("maxColorR")
	maxColor_ : MaxColorPlug = PlugDescriptor("maxColor")
	minAlpha_ : MinAlphaPlug = PlugDescriptor("minAlpha")
	minColorB_ : MinColorBPlug = PlugDescriptor("minColorB")
	minColorG_ : MinColorGPlug = PlugDescriptor("minColorG")
	minColorR_ : MinColorRPlug = PlugDescriptor("minColorR")
	minColor_ : MinColorPlug = PlugDescriptor("minColor")
	scaleRgba_ : ScaleRgbaPlug = PlugDescriptor("scaleRgba")
	sharedVertices_ : SharedVerticesPlug = PlugDescriptor("sharedVertices")
	useFaceNormals_ : UseFaceNormalsPlug = PlugDescriptor("useFaceNormals")

	# node attributes

	typeName = "vertexBakeSet"
	apiTypeInt = 473
	apiTypeStr = "kVertexBakeSet"
	typeIdInt = 1447182667
	MFnCls = om.MFnSet
	nodeLeafClassAttrs = ["alphaBlending", "bakeColor", "clampMax", "clampMin", "colorBlending", "colorSetName", "maxAlpha", "maxColorB", "maxColorG", "maxColorR", "maxColor", "minAlpha", "minColorB", "minColorG", "minColorR", "minColor", "scaleRgba", "sharedVertices", "useFaceNormals"]
	nodeLeafPlugs = ["alphaBlending", "bakeColor", "clampMax", "clampMin", "colorBlending", "colorSetName", "maxAlpha", "maxColor", "minAlpha", "minColor", "scaleRgba", "sharedVertices", "useFaceNormals"]
	pass

