

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Transform = Catalogue.Transform
else:
	from .. import retriever
	Transform = retriever.getNodeCls("Transform")
	assert Transform

# add node doc



# region plug type defs

# endregion


# define node class
class TextureDeformerHandle(Transform):

	# node attributes

	typeName = "textureDeformerHandle"
	apiTypeInt = 344
	apiTypeStr = "kTextureDeformerHandle"
	typeIdInt = 1413761092
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

