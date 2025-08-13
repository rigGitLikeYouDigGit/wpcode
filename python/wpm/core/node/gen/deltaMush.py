

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
assert WeightGeometryFilter
if T.TYPE_CHECKING:
	from .. import WeightGeometryFilter

# add node doc



# region plug type defs
class CacheBindPositionsPlug(Plug):
	parent : CachePlug = PlugDescriptor("cache")
	node : DeltaMush = None
	pass
class CacheDisplacementsPlug(Plug):
	parent : CachePlug = PlugDescriptor("cache")
	node : DeltaMush = None
	pass
class CacheFramesPlug(Plug):
	parent : CachePlug = PlugDescriptor("cache")
	node : DeltaMush = None
	pass
class CachePinBorderVerticesPlug(Plug):
	parent : CachePlug = PlugDescriptor("cache")
	node : DeltaMush = None
	pass
class CacheSmoothingAlgorithmPlug(Plug):
	parent : CachePlug = PlugDescriptor("cache")
	node : DeltaMush = None
	pass
class CacheSmoothingIterationsPlug(Plug):
	parent : CachePlug = PlugDescriptor("cache")
	node : DeltaMush = None
	pass
class CacheSmoothingStepPlug(Plug):
	parent : CachePlug = PlugDescriptor("cache")
	node : DeltaMush = None
	pass
class CachePlug(Plug):
	cacheBindPositions_ : CacheBindPositionsPlug = PlugDescriptor("cacheBindPositions")
	cbp_ : CacheBindPositionsPlug = PlugDescriptor("cacheBindPositions")
	cacheDisplacements_ : CacheDisplacementsPlug = PlugDescriptor("cacheDisplacements")
	cdis_ : CacheDisplacementsPlug = PlugDescriptor("cacheDisplacements")
	cacheFrames_ : CacheFramesPlug = PlugDescriptor("cacheFrames")
	cfrm_ : CacheFramesPlug = PlugDescriptor("cacheFrames")
	cachePinBorderVertices_ : CachePinBorderVerticesPlug = PlugDescriptor("cachePinBorderVertices")
	cpbv_ : CachePinBorderVerticesPlug = PlugDescriptor("cachePinBorderVertices")
	cacheSmoothingAlgorithm_ : CacheSmoothingAlgorithmPlug = PlugDescriptor("cacheSmoothingAlgorithm")
	csa_ : CacheSmoothingAlgorithmPlug = PlugDescriptor("cacheSmoothingAlgorithm")
	cacheSmoothingIterations_ : CacheSmoothingIterationsPlug = PlugDescriptor("cacheSmoothingIterations")
	csi_ : CacheSmoothingIterationsPlug = PlugDescriptor("cacheSmoothingIterations")
	cacheSmoothingStep_ : CacheSmoothingStepPlug = PlugDescriptor("cacheSmoothingStep")
	css_ : CacheSmoothingStepPlug = PlugDescriptor("cacheSmoothingStep")
	node : DeltaMush = None
	pass
class CacheSetupPlug(Plug):
	node : DeltaMush = None
	pass
class DisplacementPlug(Plug):
	node : DeltaMush = None
	pass
class DistanceWeightPlug(Plug):
	node : DeltaMush = None
	pass
class InwardConstraintPlug(Plug):
	node : DeltaMush = None
	pass
class OutwardConstraintPlug(Plug):
	node : DeltaMush = None
	pass
class PinBorderVerticesPlug(Plug):
	node : DeltaMush = None
	pass
class ScaleXPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : DeltaMush = None
	pass
class ScaleYPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : DeltaMush = None
	pass
class ScaleZPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : DeltaMush = None
	pass
class ScalePlug(Plug):
	scaleX_ : ScaleXPlug = PlugDescriptor("scaleX")
	sx_ : ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_ : ScaleYPlug = PlugDescriptor("scaleY")
	sy_ : ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_ : ScaleZPlug = PlugDescriptor("scaleZ")
	sz_ : ScaleZPlug = PlugDescriptor("scaleZ")
	node : DeltaMush = None
	pass
class SmoothingAlgorithmPlug(Plug):
	node : DeltaMush = None
	pass
class SmoothingIterationsPlug(Plug):
	node : DeltaMush = None
	pass
class SmoothingStepPlug(Plug):
	node : DeltaMush = None
	pass
# endregion


# define node class
class DeltaMush(WeightGeometryFilter):
	cacheBindPositions_ : CacheBindPositionsPlug = PlugDescriptor("cacheBindPositions")
	cacheDisplacements_ : CacheDisplacementsPlug = PlugDescriptor("cacheDisplacements")
	cacheFrames_ : CacheFramesPlug = PlugDescriptor("cacheFrames")
	cachePinBorderVertices_ : CachePinBorderVerticesPlug = PlugDescriptor("cachePinBorderVertices")
	cacheSmoothingAlgorithm_ : CacheSmoothingAlgorithmPlug = PlugDescriptor("cacheSmoothingAlgorithm")
	cacheSmoothingIterations_ : CacheSmoothingIterationsPlug = PlugDescriptor("cacheSmoothingIterations")
	cacheSmoothingStep_ : CacheSmoothingStepPlug = PlugDescriptor("cacheSmoothingStep")
	cache_ : CachePlug = PlugDescriptor("cache")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	displacement_ : DisplacementPlug = PlugDescriptor("displacement")
	distanceWeight_ : DistanceWeightPlug = PlugDescriptor("distanceWeight")
	inwardConstraint_ : InwardConstraintPlug = PlugDescriptor("inwardConstraint")
	outwardConstraint_ : OutwardConstraintPlug = PlugDescriptor("outwardConstraint")
	pinBorderVertices_ : PinBorderVerticesPlug = PlugDescriptor("pinBorderVertices")
	scaleX_ : ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_ : ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_ : ScaleZPlug = PlugDescriptor("scaleZ")
	scale_ : ScalePlug = PlugDescriptor("scale")
	smoothingAlgorithm_ : SmoothingAlgorithmPlug = PlugDescriptor("smoothingAlgorithm")
	smoothingIterations_ : SmoothingIterationsPlug = PlugDescriptor("smoothingIterations")
	smoothingStep_ : SmoothingStepPlug = PlugDescriptor("smoothingStep")

	# node attributes

	typeName = "deltaMush"
	apiTypeInt = 350
	apiTypeStr = "kDeltaMush"
	typeIdInt = 1145853005
	MFnCls = om.MFnGeometryFilter
	pass

