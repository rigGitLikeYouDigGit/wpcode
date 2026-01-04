

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Shape = Catalogue.Shape
else:
	from .. import retriever
	Shape = retriever.getNodeCls("Shape")
	assert Shape

# add node doc



# region plug type defs
class OriginXPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : SoftModHandle = None
	pass
class OriginYPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : SoftModHandle = None
	pass
class OriginZPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : SoftModHandle = None
	pass
class OriginPlug(Plug):
	originX_ : OriginXPlug = PlugDescriptor("originX")
	ox_ : OriginXPlug = PlugDescriptor("originX")
	originY_ : OriginYPlug = PlugDescriptor("originY")
	oy_ : OriginYPlug = PlugDescriptor("originY")
	originZ_ : OriginZPlug = PlugDescriptor("originZ")
	oz_ : OriginZPlug = PlugDescriptor("originZ")
	node : SoftModHandle = None
	pass
class PostWeightedMatrixTransformPlug(Plug):
	parent : SoftModTransformsPlug = PlugDescriptor("softModTransforms")
	node : SoftModHandle = None
	pass
class PreWeightedMatrixTransformPlug(Plug):
	parent : SoftModTransformsPlug = PlugDescriptor("softModTransforms")
	node : SoftModHandle = None
	pass
class WeightedMatrixTransformPlug(Plug):
	parent : SoftModTransformsPlug = PlugDescriptor("softModTransforms")
	node : SoftModHandle = None
	pass
class SoftModTransformsPlug(Plug):
	postWeightedMatrixTransform_ : PostWeightedMatrixTransformPlug = PlugDescriptor("postWeightedMatrixTransform")
	post_ : PostWeightedMatrixTransformPlug = PlugDescriptor("postWeightedMatrixTransform")
	preWeightedMatrixTransform_ : PreWeightedMatrixTransformPlug = PlugDescriptor("preWeightedMatrixTransform")
	pre_ : PreWeightedMatrixTransformPlug = PlugDescriptor("preWeightedMatrixTransform")
	weightedMatrixTransform_ : WeightedMatrixTransformPlug = PlugDescriptor("weightedMatrixTransform")
	wt_ : WeightedMatrixTransformPlug = PlugDescriptor("weightedMatrixTransform")
	node : SoftModHandle = None
	pass
class WeightedNodePlug(Plug):
	node : SoftModHandle = None
	pass
# endregion


# define node class
class SoftModHandle(Shape):
	originX_ : OriginXPlug = PlugDescriptor("originX")
	originY_ : OriginYPlug = PlugDescriptor("originY")
	originZ_ : OriginZPlug = PlugDescriptor("originZ")
	origin_ : OriginPlug = PlugDescriptor("origin")
	postWeightedMatrixTransform_ : PostWeightedMatrixTransformPlug = PlugDescriptor("postWeightedMatrixTransform")
	preWeightedMatrixTransform_ : PreWeightedMatrixTransformPlug = PlugDescriptor("preWeightedMatrixTransform")
	weightedMatrixTransform_ : WeightedMatrixTransformPlug = PlugDescriptor("weightedMatrixTransform")
	softModTransforms_ : SoftModTransformsPlug = PlugDescriptor("softModTransforms")
	weightedNode_ : WeightedNodePlug = PlugDescriptor("weightedNode")

	# node attributes

	typeName = "softModHandle"
	typeIdInt = 1179865928
	nodeLeafClassAttrs = ["originX", "originY", "originZ", "origin", "postWeightedMatrixTransform", "preWeightedMatrixTransform", "weightedMatrixTransform", "softModTransforms", "weightedNode"]
	nodeLeafPlugs = ["origin", "softModTransforms", "weightedNode"]
	pass

