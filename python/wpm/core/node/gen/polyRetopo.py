

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class AnisotropyPlug(Plug):
	node : PolyRetopo = None
	pass
class CurveInfluenceDirectionPlug(Plug):
	node : PolyRetopo = None
	pass
class CurveSingularitySeparationPlug(Plug):
	node : PolyRetopo = None
	pass
class FaceUniformityPlug(Plug):
	node : PolyRetopo = None
	pass
class FeatureTagsPlug(Plug):
	node : PolyRetopo = None
	pass
class PreprocessMeshPlug(Plug):
	node : PolyRetopo = None
	pass
class PreserveHardEdgesPlug(Plug):
	node : PolyRetopo = None
	pass
class TargetEdgeDeviationPlug(Plug):
	node : PolyRetopo = None
	pass
class TargetFaceCountPlug(Plug):
	node : PolyRetopo = None
	pass
class TargetFaceCountTolerancePlug(Plug):
	node : PolyRetopo = None
	pass
class TopologyRegularityPlug(Plug):
	node : PolyRetopo = None
	pass
# endregion


# define node class
class PolyRetopo(PolyModifier):
	anisotropy_ : AnisotropyPlug = PlugDescriptor("anisotropy")
	curveInfluenceDirection_ : CurveInfluenceDirectionPlug = PlugDescriptor("curveInfluenceDirection")
	curveSingularitySeparation_ : CurveSingularitySeparationPlug = PlugDescriptor("curveSingularitySeparation")
	faceUniformity_ : FaceUniformityPlug = PlugDescriptor("faceUniformity")
	featureTags_ : FeatureTagsPlug = PlugDescriptor("featureTags")
	preprocessMesh_ : PreprocessMeshPlug = PlugDescriptor("preprocessMesh")
	preserveHardEdges_ : PreserveHardEdgesPlug = PlugDescriptor("preserveHardEdges")
	targetEdgeDeviation_ : TargetEdgeDeviationPlug = PlugDescriptor("targetEdgeDeviation")
	targetFaceCount_ : TargetFaceCountPlug = PlugDescriptor("targetFaceCount")
	targetFaceCountTolerance_ : TargetFaceCountTolerancePlug = PlugDescriptor("targetFaceCountTolerance")
	topologyRegularity_ : TopologyRegularityPlug = PlugDescriptor("topologyRegularity")

	# node attributes

	typeName = "polyRetopo"
	typeIdInt = 1347569229
	pass

