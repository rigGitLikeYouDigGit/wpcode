

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ImageSource = Catalogue.ImageSource
else:
	from .. import retriever
	ImageSource = retriever.getNodeCls("ImageSource")
	assert ImageSource

# add node doc



# region plug type defs
class CameraPlug(Plug):
	node : RenderedImageSource = None
	pass
class ImageSourcePlug(Plug):
	node : RenderedImageSource = None
	pass
class RenderLayerPlug(Plug):
	node : RenderedImageSource = None
	pass
# endregion


# define node class
class RenderedImageSource(ImageSource):
	camera_ : CameraPlug = PlugDescriptor("camera")
	imageSource_ : ImageSourcePlug = PlugDescriptor("imageSource")
	renderLayer_ : RenderLayerPlug = PlugDescriptor("renderLayer")

	# node attributes

	typeName = "renderedImageSource"
	apiTypeInt = 790
	apiTypeStr = "kRenderedImageSource"
	typeIdInt = 1380141395
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["camera", "imageSource", "renderLayer"]
	nodeLeafPlugs = ["camera", "imageSource", "renderLayer"]
	pass

