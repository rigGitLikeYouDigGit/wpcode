

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
class AngleInterpolationPlug(Plug):
	node : SoftMod = None
	pass
class BindPreMatrixPlug(Plug):
	node : SoftMod = None
	pass
class DistanceCachePlug(Plug):
	node : SoftMod = None
	pass
class DistanceCacheDirtyPlug(Plug):
	node : SoftMod = None
	pass
class FalloffAroundSelectionPlug(Plug):
	node : SoftMod = None
	pass
class FalloffCenterXPlug(Plug):
	parent : FalloffCenterPlug = PlugDescriptor("falloffCenter")
	node : SoftMod = None
	pass
class FalloffCenterYPlug(Plug):
	parent : FalloffCenterPlug = PlugDescriptor("falloffCenter")
	node : SoftMod = None
	pass
class FalloffCenterZPlug(Plug):
	parent : FalloffCenterPlug = PlugDescriptor("falloffCenter")
	node : SoftMod = None
	pass
class FalloffCenterPlug(Plug):
	falloffCenterX_ : FalloffCenterXPlug = PlugDescriptor("falloffCenterX")
	fcx_ : FalloffCenterXPlug = PlugDescriptor("falloffCenterX")
	falloffCenterY_ : FalloffCenterYPlug = PlugDescriptor("falloffCenterY")
	fcy_ : FalloffCenterYPlug = PlugDescriptor("falloffCenterY")
	falloffCenterZ_ : FalloffCenterZPlug = PlugDescriptor("falloffCenterZ")
	fcz_ : FalloffCenterZPlug = PlugDescriptor("falloffCenterZ")
	node : SoftMod = None
	pass
class FalloffCurve_FloatValuePlug(Plug):
	parent : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	node : SoftMod = None
	pass
class FalloffCurve_InterpPlug(Plug):
	parent : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	node : SoftMod = None
	pass
class FalloffCurve_PositionPlug(Plug):
	parent : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	node : SoftMod = None
	pass
class FalloffCurvePlug(Plug):
	falloffCurve_FloatValue_ : FalloffCurve_FloatValuePlug = PlugDescriptor("falloffCurve_FloatValue")
	fcfv_ : FalloffCurve_FloatValuePlug = PlugDescriptor("falloffCurve_FloatValue")
	falloffCurve_Interp_ : FalloffCurve_InterpPlug = PlugDescriptor("falloffCurve_Interp")
	fci_ : FalloffCurve_InterpPlug = PlugDescriptor("falloffCurve_Interp")
	falloffCurve_Position_ : FalloffCurve_PositionPlug = PlugDescriptor("falloffCurve_Position")
	fcp_ : FalloffCurve_PositionPlug = PlugDescriptor("falloffCurve_Position")
	node : SoftMod = None
	pass
class FalloffInXPlug(Plug):
	node : SoftMod = None
	pass
class FalloffInYPlug(Plug):
	node : SoftMod = None
	pass
class FalloffInZPlug(Plug):
	node : SoftMod = None
	pass
class FalloffMaskingPlug(Plug):
	node : SoftMod = None
	pass
class FalloffModePlug(Plug):
	node : SoftMod = None
	pass
class FalloffRadiusPlug(Plug):
	node : SoftMod = None
	pass
class FastFalloffCenterPlug(Plug):
	node : SoftMod = None
	pass
class GeomMatrixPlug(Plug):
	node : SoftMod = None
	pass
class InfluenceMatrixPlug(Plug):
	node : SoftMod = None
	pass
class LimitCacheUpdatesPlug(Plug):
	node : SoftMod = None
	pass
class MatrixPlug(Plug):
	node : SoftMod = None
	pass
class PercentResolutionPlug(Plug):
	node : SoftMod = None
	pass
class RelativePlug(Plug):
	node : SoftMod = None
	pass
class PostMatrixPlug(Plug):
	parent : SoftModXformsPlug = PlugDescriptor("softModXforms")
	node : SoftMod = None
	pass
class PreMatrixPlug(Plug):
	parent : SoftModXformsPlug = PlugDescriptor("softModXforms")
	node : SoftMod = None
	pass
class WeightedMatrixPlug(Plug):
	parent : SoftModXformsPlug = PlugDescriptor("softModXforms")
	node : SoftMod = None
	pass
class SoftModXformsPlug(Plug):
	postMatrix_ : PostMatrixPlug = PlugDescriptor("postMatrix")
	post_ : PostMatrixPlug = PlugDescriptor("postMatrix")
	preMatrix_ : PreMatrixPlug = PlugDescriptor("preMatrix")
	pre_ : PreMatrixPlug = PlugDescriptor("preMatrix")
	weightedMatrix_ : WeightedMatrixPlug = PlugDescriptor("weightedMatrix")
	wt_ : WeightedMatrixPlug = PlugDescriptor("weightedMatrix")
	node : SoftMod = None
	pass
class UseDistanceCachePlug(Plug):
	node : SoftMod = None
	pass
class UsePartialResolutionPlug(Plug):
	node : SoftMod = None
	pass
class WeightedCompensationMatrixPlug(Plug):
	node : SoftMod = None
	pass
# endregion


# define node class
class SoftMod(WeightGeometryFilter):
	angleInterpolation_ : AngleInterpolationPlug = PlugDescriptor("angleInterpolation")
	bindPreMatrix_ : BindPreMatrixPlug = PlugDescriptor("bindPreMatrix")
	distanceCache_ : DistanceCachePlug = PlugDescriptor("distanceCache")
	distanceCacheDirty_ : DistanceCacheDirtyPlug = PlugDescriptor("distanceCacheDirty")
	falloffAroundSelection_ : FalloffAroundSelectionPlug = PlugDescriptor("falloffAroundSelection")
	falloffCenterX_ : FalloffCenterXPlug = PlugDescriptor("falloffCenterX")
	falloffCenterY_ : FalloffCenterYPlug = PlugDescriptor("falloffCenterY")
	falloffCenterZ_ : FalloffCenterZPlug = PlugDescriptor("falloffCenterZ")
	falloffCenter_ : FalloffCenterPlug = PlugDescriptor("falloffCenter")
	falloffCurve_FloatValue_ : FalloffCurve_FloatValuePlug = PlugDescriptor("falloffCurve_FloatValue")
	falloffCurve_Interp_ : FalloffCurve_InterpPlug = PlugDescriptor("falloffCurve_Interp")
	falloffCurve_Position_ : FalloffCurve_PositionPlug = PlugDescriptor("falloffCurve_Position")
	falloffCurve_ : FalloffCurvePlug = PlugDescriptor("falloffCurve")
	falloffInX_ : FalloffInXPlug = PlugDescriptor("falloffInX")
	falloffInY_ : FalloffInYPlug = PlugDescriptor("falloffInY")
	falloffInZ_ : FalloffInZPlug = PlugDescriptor("falloffInZ")
	falloffMasking_ : FalloffMaskingPlug = PlugDescriptor("falloffMasking")
	falloffMode_ : FalloffModePlug = PlugDescriptor("falloffMode")
	falloffRadius_ : FalloffRadiusPlug = PlugDescriptor("falloffRadius")
	fastFalloffCenter_ : FastFalloffCenterPlug = PlugDescriptor("fastFalloffCenter")
	geomMatrix_ : GeomMatrixPlug = PlugDescriptor("geomMatrix")
	influenceMatrix_ : InfluenceMatrixPlug = PlugDescriptor("influenceMatrix")
	limitCacheUpdates_ : LimitCacheUpdatesPlug = PlugDescriptor("limitCacheUpdates")
	matrix_ : MatrixPlug = PlugDescriptor("matrix")
	percentResolution_ : PercentResolutionPlug = PlugDescriptor("percentResolution")
	relative_ : RelativePlug = PlugDescriptor("relative")
	postMatrix_ : PostMatrixPlug = PlugDescriptor("postMatrix")
	preMatrix_ : PreMatrixPlug = PlugDescriptor("preMatrix")
	weightedMatrix_ : WeightedMatrixPlug = PlugDescriptor("weightedMatrix")
	softModXforms_ : SoftModXformsPlug = PlugDescriptor("softModXforms")
	useDistanceCache_ : UseDistanceCachePlug = PlugDescriptor("useDistanceCache")
	usePartialResolution_ : UsePartialResolutionPlug = PlugDescriptor("usePartialResolution")
	weightedCompensationMatrix_ : WeightedCompensationMatrixPlug = PlugDescriptor("weightedCompensationMatrix")

	# node attributes

	typeName = "softMod"
	apiTypeInt = 252
	apiTypeStr = "kSoftMod"
	typeIdInt = 1179865932
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["angleInterpolation", "bindPreMatrix", "distanceCache", "distanceCacheDirty", "falloffAroundSelection", "falloffCenterX", "falloffCenterY", "falloffCenterZ", "falloffCenter", "falloffCurve_FloatValue", "falloffCurve_Interp", "falloffCurve_Position", "falloffCurve", "falloffInX", "falloffInY", "falloffInZ", "falloffMasking", "falloffMode", "falloffRadius", "fastFalloffCenter", "geomMatrix", "influenceMatrix", "limitCacheUpdates", "matrix", "percentResolution", "relative", "postMatrix", "preMatrix", "weightedMatrix", "softModXforms", "useDistanceCache", "usePartialResolution", "weightedCompensationMatrix"]
	nodeLeafPlugs = ["angleInterpolation", "bindPreMatrix", "distanceCache", "distanceCacheDirty", "falloffAroundSelection", "falloffCenter", "falloffCurve", "falloffInX", "falloffInY", "falloffInZ", "falloffMasking", "falloffMode", "falloffRadius", "fastFalloffCenter", "geomMatrix", "influenceMatrix", "limitCacheUpdates", "matrix", "percentResolution", "relative", "softModXforms", "useDistanceCache", "usePartialResolution", "weightedCompensationMatrix"]
	pass

