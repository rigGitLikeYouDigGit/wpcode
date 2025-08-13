

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ImageSource = retriever.getNodeCls("ImageSource")
assert ImageSource
if T.TYPE_CHECKING:
	from .. import ImageSource

# add node doc



# region plug type defs
class AlphaPlug(Plug):
	node : RenderTarget = None
	pass
class CameraPlug(Plug):
	node : RenderTarget = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RenderTarget = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RenderTarget = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RenderTarget = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : RenderTarget = None
	pass
class ColorProfilePlug(Plug):
	node : RenderTarget = None
	pass
class FrameBufferOverridePlug(Plug):
	node : RenderTarget = None
	pass
class FrameBufferTypePlug(Plug):
	node : RenderTarget = None
	pass
class HeightPlug(Plug):
	node : RenderTarget = None
	pass
class NumberOfChannelsPlug(Plug):
	node : RenderTarget = None
	pass
class RenderLayerPlug(Plug):
	node : RenderTarget = None
	pass
class RenderablePlug(Plug):
	node : RenderTarget = None
	pass
class RendererPlug(Plug):
	node : RenderTarget = None
	pass
class RenderingOverridePlug(Plug):
	node : RenderTarget = None
	pass
class ResolutionOverridePlug(Plug):
	node : RenderTarget = None
	pass
class WidthPlug(Plug):
	node : RenderTarget = None
	pass
# endregion


# define node class
class RenderTarget(ImageSource):
	alpha_ : AlphaPlug = PlugDescriptor("alpha")
	camera_ : CameraPlug = PlugDescriptor("camera")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	colorProfile_ : ColorProfilePlug = PlugDescriptor("colorProfile")
	frameBufferOverride_ : FrameBufferOverridePlug = PlugDescriptor("frameBufferOverride")
	frameBufferType_ : FrameBufferTypePlug = PlugDescriptor("frameBufferType")
	height_ : HeightPlug = PlugDescriptor("height")
	numberOfChannels_ : NumberOfChannelsPlug = PlugDescriptor("numberOfChannels")
	renderLayer_ : RenderLayerPlug = PlugDescriptor("renderLayer")
	renderable_ : RenderablePlug = PlugDescriptor("renderable")
	renderer_ : RendererPlug = PlugDescriptor("renderer")
	renderingOverride_ : RenderingOverridePlug = PlugDescriptor("renderingOverride")
	resolutionOverride_ : ResolutionOverridePlug = PlugDescriptor("resolutionOverride")
	width_ : WidthPlug = PlugDescriptor("width")

	# node attributes

	typeName = "renderTarget"
	apiTypeInt = 789
	apiTypeStr = "kRenderTarget"
	typeIdInt = 1380865095
	MFnCls = om.MFnDependencyNode
	pass

