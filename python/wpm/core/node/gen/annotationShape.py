

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
DimensionShape = retriever.getNodeCls("DimensionShape")
assert DimensionShape
if T.TYPE_CHECKING:
	from .. import DimensionShape

# add node doc



# region plug type defs
class DagObjectMatrixPlug(Plug):
	node : AnnotationShape = None
	pass
class DisplayArrowPlug(Plug):
	node : AnnotationShape = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : AnnotationShape = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : AnnotationShape = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : AnnotationShape = None
	pass
class PositionPlug(Plug):
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	tpx_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	tpy_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	tpz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : AnnotationShape = None
	pass
class TextPlug(Plug):
	node : AnnotationShape = None
	pass
# endregion


# define node class
class AnnotationShape(DimensionShape):
	dagObjectMatrix_ : DagObjectMatrixPlug = PlugDescriptor("dagObjectMatrix")
	displayArrow_ : DisplayArrowPlug = PlugDescriptor("displayArrow")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	text_ : TextPlug = PlugDescriptor("text")

	# node attributes

	typeName = "annotationShape"
	typeIdInt = 1095650899
	pass

