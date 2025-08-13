

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class ClampedPlug(Plug):
	node : PolyColorPerVertex = None
	pass
class VertexColorRGBPlug(Plug):
	parent : VertexColorPlug = PlugDescriptor("vertexColor")
	vertexColorB_ : VertexColorBPlug = PlugDescriptor("vertexColorB")
	vxcb_ : VertexColorBPlug = PlugDescriptor("vertexColorB")
	vertexColorG_ : VertexColorGPlug = PlugDescriptor("vertexColorG")
	vxcg_ : VertexColorGPlug = PlugDescriptor("vertexColorG")
	vertexColorR_ : VertexColorRPlug = PlugDescriptor("vertexColorR")
	vxcr_ : VertexColorRPlug = PlugDescriptor("vertexColorR")
	node : PolyColorPerVertex = None
	pass
class VertexFaceColorRGBPlug(Plug):
	parent : VertexFaceColorPlug = PlugDescriptor("vertexFaceColor")
	vertexFaceColorB_ : VertexFaceColorBPlug = PlugDescriptor("vertexFaceColorB")
	vfcb_ : VertexFaceColorBPlug = PlugDescriptor("vertexFaceColorB")
	vertexFaceColorG_ : VertexFaceColorGPlug = PlugDescriptor("vertexFaceColorG")
	vfcg_ : VertexFaceColorGPlug = PlugDescriptor("vertexFaceColorG")
	vertexFaceColorR_ : VertexFaceColorRPlug = PlugDescriptor("vertexFaceColorR")
	vfcr_ : VertexFaceColorRPlug = PlugDescriptor("vertexFaceColorR")
	node : PolyColorPerVertex = None
	pass
class VertexFaceColorPlug(Plug):
	parent : VertexColorPlug = PlugDescriptor("vertexColor")
	vertexFaceAlpha_ : VertexFaceAlphaPlug = PlugDescriptor("vertexFaceAlpha")
	vfal_ : VertexFaceAlphaPlug = PlugDescriptor("vertexFaceAlpha")
	vertexFaceColorRGB_ : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	frgb_ : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	node : PolyColorPerVertex = None
	pass
class VertexColorPlug(Plug):
	parent : ColorPerVertexPlug = PlugDescriptor("colorPerVertex")
	vertexAlpha_ : VertexAlphaPlug = PlugDescriptor("vertexAlpha")
	vxal_ : VertexAlphaPlug = PlugDescriptor("vertexAlpha")
	vertexColorRGB_ : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	vrgb_ : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	vertexFaceColor_ : VertexFaceColorPlug = PlugDescriptor("vertexFaceColor")
	vfcl_ : VertexFaceColorPlug = PlugDescriptor("vertexFaceColor")
	node : PolyColorPerVertex = None
	pass
class ColorPerVertexPlug(Plug):
	vertexColor_ : VertexColorPlug = PlugDescriptor("vertexColor")
	vclr_ : VertexColorPlug = PlugDescriptor("vertexColor")
	node : PolyColorPerVertex = None
	pass
class ColorSetNamePlug(Plug):
	node : PolyColorPerVertex = None
	pass
class RepresentationPlug(Plug):
	node : PolyColorPerVertex = None
	pass
class VertexAlphaPlug(Plug):
	parent : VertexColorPlug = PlugDescriptor("vertexColor")
	node : PolyColorPerVertex = None
	pass
class VertexColorBPlug(Plug):
	parent : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	node : PolyColorPerVertex = None
	pass
class VertexColorGPlug(Plug):
	parent : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	node : PolyColorPerVertex = None
	pass
class VertexColorRPlug(Plug):
	parent : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	node : PolyColorPerVertex = None
	pass
class VertexFaceAlphaPlug(Plug):
	parent : VertexFaceColorPlug = PlugDescriptor("vertexFaceColor")
	node : PolyColorPerVertex = None
	pass
class VertexFaceColorBPlug(Plug):
	parent : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	node : PolyColorPerVertex = None
	pass
class VertexFaceColorGPlug(Plug):
	parent : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	node : PolyColorPerVertex = None
	pass
class VertexFaceColorRPlug(Plug):
	parent : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	node : PolyColorPerVertex = None
	pass
# endregion


# define node class
class PolyColorPerVertex(PolyModifier):
	clamped_ : ClampedPlug = PlugDescriptor("clamped")
	vertexColorRGB_ : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	vertexFaceColorRGB_ : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	vertexFaceColor_ : VertexFaceColorPlug = PlugDescriptor("vertexFaceColor")
	vertexColor_ : VertexColorPlug = PlugDescriptor("vertexColor")
	colorPerVertex_ : ColorPerVertexPlug = PlugDescriptor("colorPerVertex")
	colorSetName_ : ColorSetNamePlug = PlugDescriptor("colorSetName")
	representation_ : RepresentationPlug = PlugDescriptor("representation")
	vertexAlpha_ : VertexAlphaPlug = PlugDescriptor("vertexAlpha")
	vertexColorB_ : VertexColorBPlug = PlugDescriptor("vertexColorB")
	vertexColorG_ : VertexColorGPlug = PlugDescriptor("vertexColorG")
	vertexColorR_ : VertexColorRPlug = PlugDescriptor("vertexColorR")
	vertexFaceAlpha_ : VertexFaceAlphaPlug = PlugDescriptor("vertexFaceAlpha")
	vertexFaceColorB_ : VertexFaceColorBPlug = PlugDescriptor("vertexFaceColorB")
	vertexFaceColorG_ : VertexFaceColorGPlug = PlugDescriptor("vertexFaceColorG")
	vertexFaceColorR_ : VertexFaceColorRPlug = PlugDescriptor("vertexFaceColorR")

	# node attributes

	typeName = "polyColorPerVertex"
	apiTypeInt = 735
	apiTypeStr = "kPolyColorPerVertex"
	typeIdInt = 1346588758
	MFnCls = om.MFnDependencyNode
	pass

