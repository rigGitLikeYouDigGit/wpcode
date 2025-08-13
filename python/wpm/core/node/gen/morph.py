

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
class CacheSetupPlug(Plug):
	node : Morph = None
	pass
class ComponentLookupPlug(Plug):
	parent : ComponentLookupListPlug = PlugDescriptor("componentLookupList")
	node : Morph = None
	pass
class ComponentLookupListPlug(Plug):
	componentLookup_ : ComponentLookupPlug = PlugDescriptor("componentLookup")
	clkp_ : ComponentLookupPlug = PlugDescriptor("componentLookup")
	node : Morph = None
	pass
class InwardConstraintPlug(Plug):
	node : Morph = None
	pass
class MirrorDirectionPlug(Plug):
	node : Morph = None
	pass
class MorphModePlug(Plug):
	node : Morph = None
	pass
class MorphSpacePlug(Plug):
	node : Morph = None
	pass
class MorphTargetPlug(Plug):
	node : Morph = None
	pass
class NeighborBiasPlug(Plug):
	node : Morph = None
	pass
class NeighborExponentPlug(Plug):
	node : Morph = None
	pass
class NeighborLevelPlug(Plug):
	node : Morph = None
	pass
class NormalScalePlug(Plug):
	node : Morph = None
	pass
class OriginalMorphTargetPlug(Plug):
	node : Morph = None
	pass
class OutwardConstraintPlug(Plug):
	node : Morph = None
	pass
class ScaleEnvelopePlug(Plug):
	node : Morph = None
	pass
class ScaleLevelPlug(Plug):
	node : Morph = None
	pass
class SmoothNormalsPlug(Plug):
	node : Morph = None
	pass
class TangentPlaneScalePlug(Plug):
	node : Morph = None
	pass
class TangentialDampingPlug(Plug):
	node : Morph = None
	pass
class UniformScaleWeightPlug(Plug):
	node : Morph = None
	pass
class UseComponentLookupPlug(Plug):
	node : Morph = None
	pass
class UseOriginalMorphTargetPlug(Plug):
	node : Morph = None
	pass
class UseTangentialConstraintsPlug(Plug):
	node : Morph = None
	pass
# endregion


# define node class
class Morph(WeightGeometryFilter):
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	componentLookup_ : ComponentLookupPlug = PlugDescriptor("componentLookup")
	componentLookupList_ : ComponentLookupListPlug = PlugDescriptor("componentLookupList")
	inwardConstraint_ : InwardConstraintPlug = PlugDescriptor("inwardConstraint")
	mirrorDirection_ : MirrorDirectionPlug = PlugDescriptor("mirrorDirection")
	morphMode_ : MorphModePlug = PlugDescriptor("morphMode")
	morphSpace_ : MorphSpacePlug = PlugDescriptor("morphSpace")
	morphTarget_ : MorphTargetPlug = PlugDescriptor("morphTarget")
	neighborBias_ : NeighborBiasPlug = PlugDescriptor("neighborBias")
	neighborExponent_ : NeighborExponentPlug = PlugDescriptor("neighborExponent")
	neighborLevel_ : NeighborLevelPlug = PlugDescriptor("neighborLevel")
	normalScale_ : NormalScalePlug = PlugDescriptor("normalScale")
	originalMorphTarget_ : OriginalMorphTargetPlug = PlugDescriptor("originalMorphTarget")
	outwardConstraint_ : OutwardConstraintPlug = PlugDescriptor("outwardConstraint")
	scaleEnvelope_ : ScaleEnvelopePlug = PlugDescriptor("scaleEnvelope")
	scaleLevel_ : ScaleLevelPlug = PlugDescriptor("scaleLevel")
	smoothNormals_ : SmoothNormalsPlug = PlugDescriptor("smoothNormals")
	tangentPlaneScale_ : TangentPlaneScalePlug = PlugDescriptor("tangentPlaneScale")
	tangentialDamping_ : TangentialDampingPlug = PlugDescriptor("tangentialDamping")
	uniformScaleWeight_ : UniformScaleWeightPlug = PlugDescriptor("uniformScaleWeight")
	useComponentLookup_ : UseComponentLookupPlug = PlugDescriptor("useComponentLookup")
	useOriginalMorphTarget_ : UseOriginalMorphTargetPlug = PlugDescriptor("useOriginalMorphTarget")
	useTangentialConstraints_ : UseTangentialConstraintsPlug = PlugDescriptor("useTangentialConstraints")

	# node attributes

	typeName = "morph"
	apiTypeInt = 352
	apiTypeStr = "kMorph"
	typeIdInt = 1297240136
	MFnCls = om.MFnGeometryFilter
	pass

