

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryShape = retriever.getNodeCls("GeometryShape")
assert GeometryShape
if T.TYPE_CHECKING:
	from .. import GeometryShape

# add node doc



# region plug type defs
class BoxPlug(Plug):
	node : ImplicitBox = None
	pass
class SizeXPlug(Plug):
	parent : SizePlug = PlugDescriptor("size")
	node : ImplicitBox = None
	pass
class SizeYPlug(Plug):
	parent : SizePlug = PlugDescriptor("size")
	node : ImplicitBox = None
	pass
class SizeZPlug(Plug):
	parent : SizePlug = PlugDescriptor("size")
	node : ImplicitBox = None
	pass
class SizePlug(Plug):
	sizeX_ : SizeXPlug = PlugDescriptor("sizeX")
	szx_ : SizeXPlug = PlugDescriptor("sizeX")
	sizeY_ : SizeYPlug = PlugDescriptor("sizeY")
	szy_ : SizeYPlug = PlugDescriptor("sizeY")
	sizeZ_ : SizeZPlug = PlugDescriptor("sizeZ")
	szz_ : SizeZPlug = PlugDescriptor("sizeZ")
	node : ImplicitBox = None
	pass
# endregion


# define node class
class ImplicitBox(GeometryShape):
	box_ : BoxPlug = PlugDescriptor("box")
	sizeX_ : SizeXPlug = PlugDescriptor("sizeX")
	sizeY_ : SizeYPlug = PlugDescriptor("sizeY")
	sizeZ_ : SizeZPlug = PlugDescriptor("sizeZ")
	size_ : SizePlug = PlugDescriptor("size")

	# node attributes

	typeName = "implicitBox"
	typeIdInt = 1179206232
	pass

