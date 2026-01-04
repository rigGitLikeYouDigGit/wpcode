

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifier = Catalogue.PolyModifier
else:
	from .. import retriever
	PolyModifier = retriever.getNodeCls("PolyModifier")
	assert PolyModifier

# add node doc



# region plug type defs
class BorderPlug(Plug):
	node : PolyReduce = None
	pass
class CachingReducePlug(Plug):
	node : PolyReduce = None
	pass
class ColorWeightsPlug(Plug):
	node : PolyReduce = None
	pass
class CompactnessPlug(Plug):
	node : PolyReduce = None
	pass
class DetailPlug(Plug):
	node : PolyReduce = None
	pass
class GeomWeightsPlug(Plug):
	node : PolyReduce = None
	pass
class InvertVertexWeightsPlug(Plug):
	node : PolyReduce = None
	pass
class KeepBorderPlug(Plug):
	node : PolyReduce = None
	pass
class KeepBorderWeightPlug(Plug):
	node : PolyReduce = None
	pass
class KeepColorBorderPlug(Plug):
	node : PolyReduce = None
	pass
class KeepColorBorderWeightPlug(Plug):
	node : PolyReduce = None
	pass
class KeepCreaseEdgePlug(Plug):
	node : PolyReduce = None
	pass
class KeepCreaseEdgeWeightPlug(Plug):
	node : PolyReduce = None
	pass
class KeepFaceGroupBorderPlug(Plug):
	node : PolyReduce = None
	pass
class KeepFaceGroupBorderWeightPlug(Plug):
	node : PolyReduce = None
	pass
class KeepHardEdgePlug(Plug):
	node : PolyReduce = None
	pass
class KeepHardEdgeWeightPlug(Plug):
	node : PolyReduce = None
	pass
class KeepMapBorderPlug(Plug):
	node : PolyReduce = None
	pass
class KeepMapBorderWeightPlug(Plug):
	node : PolyReduce = None
	pass
class KeepOriginalVerticesPlug(Plug):
	node : PolyReduce = None
	pass
class KeepQuadsWeightPlug(Plug):
	node : PolyReduce = None
	pass
class LinePlug(Plug):
	node : PolyReduce = None
	pass
class PercentagePlug(Plug):
	node : PolyReduce = None
	pass
class PercentageAchievedPlug(Plug):
	node : PolyReduce = None
	pass
class PreserveTopologyPlug(Plug):
	node : PolyReduce = None
	pass
class SharpnessPlug(Plug):
	node : PolyReduce = None
	pass
class SymmetryPlaneWPlug(Plug):
	parent : SymmetryPlanePlug = PlugDescriptor("symmetryPlane")
	node : PolyReduce = None
	pass
class SymmetryPlaneXPlug(Plug):
	parent : SymmetryPlanePlug = PlugDescriptor("symmetryPlane")
	node : PolyReduce = None
	pass
class SymmetryPlaneYPlug(Plug):
	parent : SymmetryPlanePlug = PlugDescriptor("symmetryPlane")
	node : PolyReduce = None
	pass
class SymmetryPlaneZPlug(Plug):
	parent : SymmetryPlanePlug = PlugDescriptor("symmetryPlane")
	node : PolyReduce = None
	pass
class SymmetryPlanePlug(Plug):
	symmetryPlaneW_ : SymmetryPlaneWPlug = PlugDescriptor("symmetryPlaneW")
	sw_ : SymmetryPlaneWPlug = PlugDescriptor("symmetryPlaneW")
	symmetryPlaneX_ : SymmetryPlaneXPlug = PlugDescriptor("symmetryPlaneX")
	sx_ : SymmetryPlaneXPlug = PlugDescriptor("symmetryPlaneX")
	symmetryPlaneY_ : SymmetryPlaneYPlug = PlugDescriptor("symmetryPlaneY")
	sy_ : SymmetryPlaneYPlug = PlugDescriptor("symmetryPlaneY")
	symmetryPlaneZ_ : SymmetryPlaneZPlug = PlugDescriptor("symmetryPlaneZ")
	sz_ : SymmetryPlaneZPlug = PlugDescriptor("symmetryPlaneZ")
	node : PolyReduce = None
	pass
class SymmetryTolerancePlug(Plug):
	node : PolyReduce = None
	pass
class TerminationPlug(Plug):
	node : PolyReduce = None
	pass
class TriangleCountPlug(Plug):
	node : PolyReduce = None
	pass
class TriangleCountAchievedPlug(Plug):
	node : PolyReduce = None
	pass
class TriangleCountInPlug(Plug):
	node : PolyReduce = None
	pass
class TriangulatePlug(Plug):
	node : PolyReduce = None
	pass
class UseVirtualSymmetryPlug(Plug):
	node : PolyReduce = None
	pass
class UvWeightsPlug(Plug):
	node : PolyReduce = None
	pass
class VersionPlug(Plug):
	node : PolyReduce = None
	pass
class VertexCountPlug(Plug):
	node : PolyReduce = None
	pass
class VertexCountAchievedPlug(Plug):
	node : PolyReduce = None
	pass
class VertexCountInPlug(Plug):
	node : PolyReduce = None
	pass
class VertexMapNamePlug(Plug):
	node : PolyReduce = None
	pass
class VertexWeightCoefficientPlug(Plug):
	node : PolyReduce = None
	pass
class VertexWeightsPlug(Plug):
	node : PolyReduce = None
	pass
class WeightCoefficientPlug(Plug):
	node : PolyReduce = None
	pass
class WeightsPlug(Plug):
	node : PolyReduce = None
	pass
# endregion


# define node class
class PolyReduce(PolyModifier):
	border_ : BorderPlug = PlugDescriptor("border")
	cachingReduce_ : CachingReducePlug = PlugDescriptor("cachingReduce")
	colorWeights_ : ColorWeightsPlug = PlugDescriptor("colorWeights")
	compactness_ : CompactnessPlug = PlugDescriptor("compactness")
	detail_ : DetailPlug = PlugDescriptor("detail")
	geomWeights_ : GeomWeightsPlug = PlugDescriptor("geomWeights")
	invertVertexWeights_ : InvertVertexWeightsPlug = PlugDescriptor("invertVertexWeights")
	keepBorder_ : KeepBorderPlug = PlugDescriptor("keepBorder")
	keepBorderWeight_ : KeepBorderWeightPlug = PlugDescriptor("keepBorderWeight")
	keepColorBorder_ : KeepColorBorderPlug = PlugDescriptor("keepColorBorder")
	keepColorBorderWeight_ : KeepColorBorderWeightPlug = PlugDescriptor("keepColorBorderWeight")
	keepCreaseEdge_ : KeepCreaseEdgePlug = PlugDescriptor("keepCreaseEdge")
	keepCreaseEdgeWeight_ : KeepCreaseEdgeWeightPlug = PlugDescriptor("keepCreaseEdgeWeight")
	keepFaceGroupBorder_ : KeepFaceGroupBorderPlug = PlugDescriptor("keepFaceGroupBorder")
	keepFaceGroupBorderWeight_ : KeepFaceGroupBorderWeightPlug = PlugDescriptor("keepFaceGroupBorderWeight")
	keepHardEdge_ : KeepHardEdgePlug = PlugDescriptor("keepHardEdge")
	keepHardEdgeWeight_ : KeepHardEdgeWeightPlug = PlugDescriptor("keepHardEdgeWeight")
	keepMapBorder_ : KeepMapBorderPlug = PlugDescriptor("keepMapBorder")
	keepMapBorderWeight_ : KeepMapBorderWeightPlug = PlugDescriptor("keepMapBorderWeight")
	keepOriginalVertices_ : KeepOriginalVerticesPlug = PlugDescriptor("keepOriginalVertices")
	keepQuadsWeight_ : KeepQuadsWeightPlug = PlugDescriptor("keepQuadsWeight")
	line_ : LinePlug = PlugDescriptor("line")
	percentage_ : PercentagePlug = PlugDescriptor("percentage")
	percentageAchieved_ : PercentageAchievedPlug = PlugDescriptor("percentageAchieved")
	preserveTopology_ : PreserveTopologyPlug = PlugDescriptor("preserveTopology")
	sharpness_ : SharpnessPlug = PlugDescriptor("sharpness")
	symmetryPlaneW_ : SymmetryPlaneWPlug = PlugDescriptor("symmetryPlaneW")
	symmetryPlaneX_ : SymmetryPlaneXPlug = PlugDescriptor("symmetryPlaneX")
	symmetryPlaneY_ : SymmetryPlaneYPlug = PlugDescriptor("symmetryPlaneY")
	symmetryPlaneZ_ : SymmetryPlaneZPlug = PlugDescriptor("symmetryPlaneZ")
	symmetryPlane_ : SymmetryPlanePlug = PlugDescriptor("symmetryPlane")
	symmetryTolerance_ : SymmetryTolerancePlug = PlugDescriptor("symmetryTolerance")
	termination_ : TerminationPlug = PlugDescriptor("termination")
	triangleCount_ : TriangleCountPlug = PlugDescriptor("triangleCount")
	triangleCountAchieved_ : TriangleCountAchievedPlug = PlugDescriptor("triangleCountAchieved")
	triangleCountIn_ : TriangleCountInPlug = PlugDescriptor("triangleCountIn")
	triangulate_ : TriangulatePlug = PlugDescriptor("triangulate")
	useVirtualSymmetry_ : UseVirtualSymmetryPlug = PlugDescriptor("useVirtualSymmetry")
	uvWeights_ : UvWeightsPlug = PlugDescriptor("uvWeights")
	version_ : VersionPlug = PlugDescriptor("version")
	vertexCount_ : VertexCountPlug = PlugDescriptor("vertexCount")
	vertexCountAchieved_ : VertexCountAchievedPlug = PlugDescriptor("vertexCountAchieved")
	vertexCountIn_ : VertexCountInPlug = PlugDescriptor("vertexCountIn")
	vertexMapName_ : VertexMapNamePlug = PlugDescriptor("vertexMapName")
	vertexWeightCoefficient_ : VertexWeightCoefficientPlug = PlugDescriptor("vertexWeightCoefficient")
	vertexWeights_ : VertexWeightsPlug = PlugDescriptor("vertexWeights")
	weightCoefficient_ : WeightCoefficientPlug = PlugDescriptor("weightCoefficient")
	weights_ : WeightsPlug = PlugDescriptor("weights")

	# node attributes

	typeName = "polyReduce"
	apiTypeInt = 770
	apiTypeStr = "kPolyReduce"
	typeIdInt = 1347568964
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["border", "cachingReduce", "colorWeights", "compactness", "detail", "geomWeights", "invertVertexWeights", "keepBorder", "keepBorderWeight", "keepColorBorder", "keepColorBorderWeight", "keepCreaseEdge", "keepCreaseEdgeWeight", "keepFaceGroupBorder", "keepFaceGroupBorderWeight", "keepHardEdge", "keepHardEdgeWeight", "keepMapBorder", "keepMapBorderWeight", "keepOriginalVertices", "keepQuadsWeight", "line", "percentage", "percentageAchieved", "preserveTopology", "sharpness", "symmetryPlaneW", "symmetryPlaneX", "symmetryPlaneY", "symmetryPlaneZ", "symmetryPlane", "symmetryTolerance", "termination", "triangleCount", "triangleCountAchieved", "triangleCountIn", "triangulate", "useVirtualSymmetry", "uvWeights", "version", "vertexCount", "vertexCountAchieved", "vertexCountIn", "vertexMapName", "vertexWeightCoefficient", "vertexWeights", "weightCoefficient", "weights"]
	nodeLeafPlugs = ["border", "cachingReduce", "colorWeights", "compactness", "detail", "geomWeights", "invertVertexWeights", "keepBorder", "keepBorderWeight", "keepColorBorder", "keepColorBorderWeight", "keepCreaseEdge", "keepCreaseEdgeWeight", "keepFaceGroupBorder", "keepFaceGroupBorderWeight", "keepHardEdge", "keepHardEdgeWeight", "keepMapBorder", "keepMapBorderWeight", "keepOriginalVertices", "keepQuadsWeight", "line", "percentage", "percentageAchieved", "preserveTopology", "sharpness", "symmetryPlane", "symmetryTolerance", "termination", "triangleCount", "triangleCountAchieved", "triangleCountIn", "triangulate", "useVirtualSymmetry", "uvWeights", "version", "vertexCount", "vertexCountAchieved", "vertexCountIn", "vertexMapName", "vertexWeightCoefficient", "vertexWeights", "weightCoefficient", "weights"]
	pass

