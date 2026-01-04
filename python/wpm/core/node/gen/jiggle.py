

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	WeightGeometryFilter = Catalogue.WeightGeometryFilter
else:
	from .. import retriever
	WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
	assert WeightGeometryFilter

# add node doc



# region plug type defs
class CacheSetupPlug(Plug):
	node : Jiggle = None
	pass
class CachedInputPositionListPlug(Plug):
	parent : CachedDataPlug = PlugDescriptor("cachedData")
	node : Jiggle = None
	pass
class CachedPositionListPlug(Plug):
	parent : CachedDataPlug = PlugDescriptor("cachedData")
	node : Jiggle = None
	pass
class CachedTimePlug(Plug):
	parent : CachedDataPlug = PlugDescriptor("cachedData")
	node : Jiggle = None
	pass
class CachedVelocityListPlug(Plug):
	parent : CachedDataPlug = PlugDescriptor("cachedData")
	node : Jiggle = None
	pass
class IsRestingPlug(Plug):
	parent : CachedDataPlug = PlugDescriptor("cachedData")
	node : Jiggle = None
	pass
class CachedDataPlug(Plug):
	parent : CachedDataListPlug = PlugDescriptor("cachedDataList")
	cachedInputPositionList_ : CachedInputPositionListPlug = PlugDescriptor("cachedInputPositionList")
	cipl_ : CachedInputPositionListPlug = PlugDescriptor("cachedInputPositionList")
	cachedPositionList_ : CachedPositionListPlug = PlugDescriptor("cachedPositionList")
	cpl_ : CachedPositionListPlug = PlugDescriptor("cachedPositionList")
	cachedTime_ : CachedTimePlug = PlugDescriptor("cachedTime")
	chti_ : CachedTimePlug = PlugDescriptor("cachedTime")
	cachedVelocityList_ : CachedVelocityListPlug = PlugDescriptor("cachedVelocityList")
	cvl_ : CachedVelocityListPlug = PlugDescriptor("cachedVelocityList")
	isResting_ : IsRestingPlug = PlugDescriptor("isResting")
	ir_ : IsRestingPlug = PlugDescriptor("isResting")
	node : Jiggle = None
	pass
class CachedDataListPlug(Plug):
	cachedData_ : CachedDataPlug = PlugDescriptor("cachedData")
	cd_ : CachedDataPlug = PlugDescriptor("cachedData")
	node : Jiggle = None
	pass
class CurrentTimePlug(Plug):
	node : Jiggle = None
	pass
class DampingPlug(Plug):
	node : Jiggle = None
	pass
class DirectionBiasPlug(Plug):
	node : Jiggle = None
	pass
class DiskCachePlug(Plug):
	node : Jiggle = None
	pass
class EnablePlug(Plug):
	node : Jiggle = None
	pass
class ForceAlongNormalPlug(Plug):
	node : Jiggle = None
	pass
class ForceOnTangentPlug(Plug):
	node : Jiggle = None
	pass
class IgnoreTransformPlug(Plug):
	node : Jiggle = None
	pass
class JiggleWeightPlug(Plug):
	node : Jiggle = None
	pass
class MotionMultiplierPlug(Plug):
	node : Jiggle = None
	pass
class StiffnessPlug(Plug):
	node : Jiggle = None
	pass
# endregion


# define node class
class Jiggle(WeightGeometryFilter):
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	cachedInputPositionList_ : CachedInputPositionListPlug = PlugDescriptor("cachedInputPositionList")
	cachedPositionList_ : CachedPositionListPlug = PlugDescriptor("cachedPositionList")
	cachedTime_ : CachedTimePlug = PlugDescriptor("cachedTime")
	cachedVelocityList_ : CachedVelocityListPlug = PlugDescriptor("cachedVelocityList")
	isResting_ : IsRestingPlug = PlugDescriptor("isResting")
	cachedData_ : CachedDataPlug = PlugDescriptor("cachedData")
	cachedDataList_ : CachedDataListPlug = PlugDescriptor("cachedDataList")
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	damping_ : DampingPlug = PlugDescriptor("damping")
	directionBias_ : DirectionBiasPlug = PlugDescriptor("directionBias")
	diskCache_ : DiskCachePlug = PlugDescriptor("diskCache")
	enable_ : EnablePlug = PlugDescriptor("enable")
	forceAlongNormal_ : ForceAlongNormalPlug = PlugDescriptor("forceAlongNormal")
	forceOnTangent_ : ForceOnTangentPlug = PlugDescriptor("forceOnTangent")
	ignoreTransform_ : IgnoreTransformPlug = PlugDescriptor("ignoreTransform")
	jiggleWeight_ : JiggleWeightPlug = PlugDescriptor("jiggleWeight")
	motionMultiplier_ : MotionMultiplierPlug = PlugDescriptor("motionMultiplier")
	stiffness_ : StiffnessPlug = PlugDescriptor("stiffness")

	# node attributes

	typeName = "jiggle"
	typeIdInt = 1246184518
	nodeLeafClassAttrs = ["cacheSetup", "cachedInputPositionList", "cachedPositionList", "cachedTime", "cachedVelocityList", "isResting", "cachedData", "cachedDataList", "currentTime", "damping", "directionBias", "diskCache", "enable", "forceAlongNormal", "forceOnTangent", "ignoreTransform", "jiggleWeight", "motionMultiplier", "stiffness"]
	nodeLeafPlugs = ["cacheSetup", "cachedDataList", "currentTime", "damping", "directionBias", "diskCache", "enable", "forceAlongNormal", "forceOnTangent", "ignoreTransform", "jiggleWeight", "motionMultiplier", "stiffness"]
	pass

