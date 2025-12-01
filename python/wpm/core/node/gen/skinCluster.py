

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryFilter = retriever.getNodeCls("GeometryFilter")
assert GeometryFilter
if T.TYPE_CHECKING:
	from .. import GeometryFilter

# add node doc



# region plug type defs
class BaseDirtyPlug(Plug):
	node : SkinCluster = None
	pass
class BasePointsPlug(Plug):
	node : SkinCluster = None
	pass
class BindMethodPlug(Plug):
	node : SkinCluster = None
	pass
class BindPosePlug(Plug):
	node : SkinCluster = None
	pass
class BindPreMatrixPlug(Plug):
	node : SkinCluster = None
	pass
class BindVolumePlug(Plug):
	node : SkinCluster = None
	pass
class BlendWeightsPlug(Plug):
	node : SkinCluster = None
	pass
class CacheSetupPlug(Plug):
	node : SkinCluster = None
	pass
class DeformUserNormalsPlug(Plug):
	node : SkinCluster = None
	pass
class DqsScaleXPlug(Plug):
	parent : DqsScalePlug = PlugDescriptor("dqsScale")
	node : SkinCluster = None
	pass
class DqsScaleYPlug(Plug):
	parent : DqsScalePlug = PlugDescriptor("dqsScale")
	node : SkinCluster = None
	pass
class DqsScaleZPlug(Plug):
	parent : DqsScalePlug = PlugDescriptor("dqsScale")
	node : SkinCluster = None
	pass
class DqsScalePlug(Plug):
	dqsScaleX_ : DqsScaleXPlug = PlugDescriptor("dqsScaleX")
	dscx_ : DqsScaleXPlug = PlugDescriptor("dqsScaleX")
	dqsScaleY_ : DqsScaleYPlug = PlugDescriptor("dqsScaleY")
	dscy_ : DqsScaleYPlug = PlugDescriptor("dqsScaleY")
	dqsScaleZ_ : DqsScaleZPlug = PlugDescriptor("dqsScaleZ")
	dscz_ : DqsScaleZPlug = PlugDescriptor("dqsScaleZ")
	node : SkinCluster = None
	pass
class DqsSupportNonRigidPlug(Plug):
	node : SkinCluster = None
	pass
class DriverPointsPlug(Plug):
	node : SkinCluster = None
	pass
class DropoffPlug(Plug):
	node : SkinCluster = None
	pass
class DropoffRatePlug(Plug):
	node : SkinCluster = None
	pass
class GeomBindPlug(Plug):
	node : SkinCluster = None
	pass
class GeomMatrixPlug(Plug):
	node : SkinCluster = None
	pass
class HeatmapFalloffPlug(Plug):
	node : SkinCluster = None
	pass
class InfluenceColorBPlug(Plug):
	parent : InfluenceColorPlug = PlugDescriptor("influenceColor")
	node : SkinCluster = None
	pass
class InfluenceColorGPlug(Plug):
	parent : InfluenceColorPlug = PlugDescriptor("influenceColor")
	node : SkinCluster = None
	pass
class InfluenceColorRPlug(Plug):
	parent : InfluenceColorPlug = PlugDescriptor("influenceColor")
	node : SkinCluster = None
	pass
class InfluenceColorPlug(Plug):
	influenceColorB_ : InfluenceColorBPlug = PlugDescriptor("influenceColorB")
	ifcb_ : InfluenceColorBPlug = PlugDescriptor("influenceColorB")
	influenceColorG_ : InfluenceColorGPlug = PlugDescriptor("influenceColorG")
	ifcg_ : InfluenceColorGPlug = PlugDescriptor("influenceColorG")
	influenceColorR_ : InfluenceColorRPlug = PlugDescriptor("influenceColorR")
	ifcr_ : InfluenceColorRPlug = PlugDescriptor("influenceColorR")
	node : SkinCluster = None
	pass
class LockWeightsPlug(Plug):
	node : SkinCluster = None
	pass
class MaintainMaxInfluencesPlug(Plug):
	node : SkinCluster = None
	pass
class MatrixPlug(Plug):
	node : SkinCluster = None
	pass
class MaxInfluencesPlug(Plug):
	node : SkinCluster = None
	pass
class NormalizeWeightsPlug(Plug):
	node : SkinCluster = None
	pass
class NurbsSamplesPlug(Plug):
	node : SkinCluster = None
	pass
class PaintArrDirtyPlug(Plug):
	node : SkinCluster = None
	pass
class PaintTransPlug(Plug):
	node : SkinCluster = None
	pass
class PaintWeightsPlug(Plug):
	node : SkinCluster = None
	pass
class PerInfluenceVertexWeightsPlug(Plug):
	parent : PerInfluenceWeightsPlug = PlugDescriptor("perInfluenceWeights")
	node : SkinCluster = None
	pass
class PerInfluenceWeightsPlug(Plug):
	perInfluenceVertexWeights_ : PerInfluenceVertexWeightsPlug = PlugDescriptor("perInfluenceVertexWeights")
	pivw_ : PerInfluenceVertexWeightsPlug = PlugDescriptor("perInfluenceVertexWeights")
	node : SkinCluster = None
	pass
class RelativeSpaceMatrixPlug(Plug):
	node : SkinCluster = None
	pass
class RelativeSpaceModePlug(Plug):
	node : SkinCluster = None
	pass
class SkinningMethodPlug(Plug):
	node : SkinCluster = None
	pass
class SmoothnessPlug(Plug):
	node : SkinCluster = None
	pass
class UseComponentsPlug(Plug):
	node : SkinCluster = None
	pass
class UseComponentsMatrixPlug(Plug):
	node : SkinCluster = None
	pass
class WeightDistributionPlug(Plug):
	node : SkinCluster = None
	pass
class WeightsPlug(Plug):
	parent : WeightListPlug = PlugDescriptor("weightList")
	node : SkinCluster = None
	pass
class WeightListPlug(Plug):
	weights_ : WeightsPlug = PlugDescriptor("weights")
	w_ : WeightsPlug = PlugDescriptor("weights")
	node : SkinCluster = None
	pass
class WtDrtyPlug(Plug):
	node : SkinCluster = None
	pass
# endregion


# define node class
class SkinCluster(GeometryFilter):
	baseDirty_ : BaseDirtyPlug = PlugDescriptor("baseDirty")
	basePoints_ : BasePointsPlug = PlugDescriptor("basePoints")
	bindMethod_ : BindMethodPlug = PlugDescriptor("bindMethod")
	bindPose_ : BindPosePlug = PlugDescriptor("bindPose")
	bindPreMatrix_ : BindPreMatrixPlug = PlugDescriptor("bindPreMatrix")
	bindVolume_ : BindVolumePlug = PlugDescriptor("bindVolume")
	blendWeights_ : BlendWeightsPlug = PlugDescriptor("blendWeights")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	deformUserNormals_ : DeformUserNormalsPlug = PlugDescriptor("deformUserNormals")
	dqsScaleX_ : DqsScaleXPlug = PlugDescriptor("dqsScaleX")
	dqsScaleY_ : DqsScaleYPlug = PlugDescriptor("dqsScaleY")
	dqsScaleZ_ : DqsScaleZPlug = PlugDescriptor("dqsScaleZ")
	dqsScale_ : DqsScalePlug = PlugDescriptor("dqsScale")
	dqsSupportNonRigid_ : DqsSupportNonRigidPlug = PlugDescriptor("dqsSupportNonRigid")
	driverPoints_ : DriverPointsPlug = PlugDescriptor("driverPoints")
	dropoff_ : DropoffPlug = PlugDescriptor("dropoff")
	dropoffRate_ : DropoffRatePlug = PlugDescriptor("dropoffRate")
	geomBind_ : GeomBindPlug = PlugDescriptor("geomBind")
	geomMatrix_ : GeomMatrixPlug = PlugDescriptor("geomMatrix")
	heatmapFalloff_ : HeatmapFalloffPlug = PlugDescriptor("heatmapFalloff")
	influenceColorB_ : InfluenceColorBPlug = PlugDescriptor("influenceColorB")
	influenceColorG_ : InfluenceColorGPlug = PlugDescriptor("influenceColorG")
	influenceColorR_ : InfluenceColorRPlug = PlugDescriptor("influenceColorR")
	influenceColor_ : InfluenceColorPlug = PlugDescriptor("influenceColor")
	lockWeights_ : LockWeightsPlug = PlugDescriptor("lockWeights")
	maintainMaxInfluences_ : MaintainMaxInfluencesPlug = PlugDescriptor("maintainMaxInfluences")
	matrix_ : MatrixPlug = PlugDescriptor("matrix")
	maxInfluences_ : MaxInfluencesPlug = PlugDescriptor("maxInfluences")
	normalizeWeights_ : NormalizeWeightsPlug = PlugDescriptor("normalizeWeights")
	nurbsSamples_ : NurbsSamplesPlug = PlugDescriptor("nurbsSamples")
	paintArrDirty_ : PaintArrDirtyPlug = PlugDescriptor("paintArrDirty")
	paintTrans_ : PaintTransPlug = PlugDescriptor("paintTrans")
	paintWeights_ : PaintWeightsPlug = PlugDescriptor("paintWeights")
	perInfluenceVertexWeights_ : PerInfluenceVertexWeightsPlug = PlugDescriptor("perInfluenceVertexWeights")
	perInfluenceWeights_ : PerInfluenceWeightsPlug = PlugDescriptor("perInfluenceWeights")
	relativeSpaceMatrix_ : RelativeSpaceMatrixPlug = PlugDescriptor("relativeSpaceMatrix")
	relativeSpaceMode_ : RelativeSpaceModePlug = PlugDescriptor("relativeSpaceMode")
	skinningMethod_ : SkinningMethodPlug = PlugDescriptor("skinningMethod")
	smoothness_ : SmoothnessPlug = PlugDescriptor("smoothness")
	useComponents_ : UseComponentsPlug = PlugDescriptor("useComponents")
	useComponentsMatrix_ : UseComponentsMatrixPlug = PlugDescriptor("useComponentsMatrix")
	weightDistribution_ : WeightDistributionPlug = PlugDescriptor("weightDistribution")
	weights_ : WeightsPlug = PlugDescriptor("weights")
	weightList_ : WeightListPlug = PlugDescriptor("weightList")
	wtDrty_ : WtDrtyPlug = PlugDescriptor("wtDrty")

	# node attributes

	typeName = "skinCluster"
	typeIdInt = 1179861836
	nodeLeafClassAttrs = ["baseDirty", "basePoints", "bindMethod", "bindPose", "bindPreMatrix", "bindVolume", "blendWeights", "cacheSetup", "deformUserNormals", "dqsScaleX", "dqsScaleY", "dqsScaleZ", "dqsScale", "dqsSupportNonRigid", "driverPoints", "dropoff", "dropoffRate", "geomBind", "geomMatrix", "heatmapFalloff", "influenceColorB", "influenceColorG", "influenceColorR", "influenceColor", "lockWeights", "maintainMaxInfluences", "matrix", "maxInfluences", "normalizeWeights", "nurbsSamples", "paintArrDirty", "paintTrans", "paintWeights", "perInfluenceVertexWeights", "perInfluenceWeights", "relativeSpaceMatrix", "relativeSpaceMode", "skinningMethod", "smoothness", "useComponents", "useComponentsMatrix", "weightDistribution", "weights", "weightList", "wtDrty"]
	nodeLeafPlugs = ["baseDirty", "basePoints", "bindMethod", "bindPose", "bindPreMatrix", "bindVolume", "blendWeights", "cacheSetup", "deformUserNormals", "dqsScale", "dqsSupportNonRigid", "driverPoints", "dropoff", "dropoffRate", "geomBind", "geomMatrix", "heatmapFalloff", "influenceColor", "lockWeights", "maintainMaxInfluences", "matrix", "maxInfluences", "normalizeWeights", "nurbsSamples", "paintArrDirty", "paintTrans", "paintWeights", "perInfluenceWeights", "relativeSpaceMatrix", "relativeSpaceMode", "skinningMethod", "smoothness", "useComponents", "useComponentsMatrix", "weightDistribution", "weightList", "wtDrty"]
	pass

