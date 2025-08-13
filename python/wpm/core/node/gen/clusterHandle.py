

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class PostWeightedMatrixTransformPlug(Plug):
	parent : ClusterTransformsPlug = PlugDescriptor("clusterTransforms")
	node : ClusterHandle = None
	pass
class PreWeightedMatrixTransformPlug(Plug):
	parent : ClusterTransformsPlug = PlugDescriptor("clusterTransforms")
	node : ClusterHandle = None
	pass
class WeightedMatrixTransformPlug(Plug):
	parent : ClusterTransformsPlug = PlugDescriptor("clusterTransforms")
	node : ClusterHandle = None
	pass
class ClusterTransformsPlug(Plug):
	postWeightedMatrixTransform_ : PostWeightedMatrixTransformPlug = PlugDescriptor("postWeightedMatrixTransform")
	post_ : PostWeightedMatrixTransformPlug = PlugDescriptor("postWeightedMatrixTransform")
	preWeightedMatrixTransform_ : PreWeightedMatrixTransformPlug = PlugDescriptor("preWeightedMatrixTransform")
	pre_ : PreWeightedMatrixTransformPlug = PlugDescriptor("preWeightedMatrixTransform")
	weightedMatrixTransform_ : WeightedMatrixTransformPlug = PlugDescriptor("weightedMatrixTransform")
	wt_ : WeightedMatrixTransformPlug = PlugDescriptor("weightedMatrixTransform")
	node : ClusterHandle = None
	pass
class OriginXPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : ClusterHandle = None
	pass
class OriginYPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : ClusterHandle = None
	pass
class OriginZPlug(Plug):
	parent : OriginPlug = PlugDescriptor("origin")
	node : ClusterHandle = None
	pass
class OriginPlug(Plug):
	originX_ : OriginXPlug = PlugDescriptor("originX")
	ox_ : OriginXPlug = PlugDescriptor("originX")
	originY_ : OriginYPlug = PlugDescriptor("originY")
	oy_ : OriginYPlug = PlugDescriptor("originY")
	originZ_ : OriginZPlug = PlugDescriptor("originZ")
	oz_ : OriginZPlug = PlugDescriptor("originZ")
	node : ClusterHandle = None
	pass
class WeightedNodePlug(Plug):
	node : ClusterHandle = None
	pass
# endregion


# define node class
class ClusterHandle(Shape):
	postWeightedMatrixTransform_ : PostWeightedMatrixTransformPlug = PlugDescriptor("postWeightedMatrixTransform")
	preWeightedMatrixTransform_ : PreWeightedMatrixTransformPlug = PlugDescriptor("preWeightedMatrixTransform")
	weightedMatrixTransform_ : WeightedMatrixTransformPlug = PlugDescriptor("weightedMatrixTransform")
	clusterTransforms_ : ClusterTransformsPlug = PlugDescriptor("clusterTransforms")
	originX_ : OriginXPlug = PlugDescriptor("originX")
	originY_ : OriginYPlug = PlugDescriptor("originY")
	originZ_ : OriginZPlug = PlugDescriptor("originZ")
	origin_ : OriginPlug = PlugDescriptor("origin")
	weightedNode_ : WeightedNodePlug = PlugDescriptor("weightedNode")

	# node attributes

	typeName = "clusterHandle"
	typeIdInt = 1178815560
	pass

