

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SurfaceShape = retriever.getNodeCls("SurfaceShape")
assert SurfaceShape
if T.TYPE_CHECKING:
	from .. import SurfaceShape

# add node doc



# region plug type defs
class BaseFaceCountPlug(Plug):
	node : Subdiv = None
	pass
class CachedPlug(Plug):
	node : Subdiv = None
	pass
class CreatePlug(Plug):
	node : Subdiv = None
	pass
class DepthPlug(Plug):
	node : Subdiv = None
	pass
class DispCreasesPlug(Plug):
	node : Subdiv = None
	pass
class DispEdgesPlug(Plug):
	node : Subdiv = None
	pass
class DispFacesPlug(Plug):
	node : Subdiv = None
	pass
class DispGeometryPlug(Plug):
	node : Subdiv = None
	pass
class DispMapsPlug(Plug):
	node : Subdiv = None
	pass
class DispResolutionPlug(Plug):
	node : Subdiv = None
	pass
class DispUVBorderPlug(Plug):
	node : Subdiv = None
	pass
class DispVerticesPlug(Plug):
	node : Subdiv = None
	pass
class DispVerticesAsLimitPointsPlug(Plug):
	node : Subdiv = None
	pass
class DisplayFilterPlug(Plug):
	node : Subdiv = None
	pass
class DisplayLevelPlug(Plug):
	node : Subdiv = None
	pass
class EdgeCreasePlug(Plug):
	node : Subdiv = None
	pass
class FaceUVIdsPlug(Plug):
	node : Subdiv = None
	pass
class FormatPlug(Plug):
	node : Subdiv = None
	pass
class LevelOneFaceCountPlug(Plug):
	node : Subdiv = None
	pass
class LocalizeLimitPointsEditPlug(Plug):
	node : Subdiv = None
	pass
class NormalsDisplayScalePlug(Plug):
	node : Subdiv = None
	pass
class OutSubdivPlug(Plug):
	node : Subdiv = None
	pass
class SampleCountPlug(Plug):
	node : Subdiv = None
	pass
class ScalingHierarchyPlug(Plug):
	node : Subdiv = None
	pass
class TextureCoordPlug(Plug):
	node : Subdiv = None
	pass
class SingleVertexXPlug(Plug):
	parent : SingleVertexPlug = PlugDescriptor("singleVertex")
	node : Subdiv = None
	pass
class SingleVertexYPlug(Plug):
	parent : SingleVertexPlug = PlugDescriptor("singleVertex")
	node : Subdiv = None
	pass
class SingleVertexZPlug(Plug):
	parent : SingleVertexPlug = PlugDescriptor("singleVertex")
	node : Subdiv = None
	pass
class SingleVertexPlug(Plug):
	parent : VertexPlug = PlugDescriptor("vertex")
	singleVertexX_ : SingleVertexXPlug = PlugDescriptor("singleVertexX")
	svx_ : SingleVertexXPlug = PlugDescriptor("singleVertexX")
	singleVertexY_ : SingleVertexYPlug = PlugDescriptor("singleVertexY")
	svy_ : SingleVertexYPlug = PlugDescriptor("singleVertexY")
	singleVertexZ_ : SingleVertexZPlug = PlugDescriptor("singleVertexZ")
	svz_ : SingleVertexZPlug = PlugDescriptor("singleVertexZ")
	node : Subdiv = None
	pass
class VertexPlug(Plug):
	singleVertex_ : SingleVertexPlug = PlugDescriptor("singleVertex")
	svt_ : SingleVertexPlug = PlugDescriptor("singleVertex")
	node : Subdiv = None
	pass
class SingleVertexTweakXPlug(Plug):
	parent : SingleVertexTweakPlug = PlugDescriptor("singleVertexTweak")
	node : Subdiv = None
	pass
class SingleVertexTweakYPlug(Plug):
	parent : SingleVertexTweakPlug = PlugDescriptor("singleVertexTweak")
	node : Subdiv = None
	pass
class SingleVertexTweakZPlug(Plug):
	parent : SingleVertexTweakPlug = PlugDescriptor("singleVertexTweak")
	node : Subdiv = None
	pass
class SingleVertexTweakPlug(Plug):
	parent : VertexTweakPlug = PlugDescriptor("vertexTweak")
	singleVertexTweakX_ : SingleVertexTweakXPlug = PlugDescriptor("singleVertexTweakX")
	stwx_ : SingleVertexTweakXPlug = PlugDescriptor("singleVertexTweakX")
	singleVertexTweakY_ : SingleVertexTweakYPlug = PlugDescriptor("singleVertexTweakY")
	stwy_ : SingleVertexTweakYPlug = PlugDescriptor("singleVertexTweakY")
	singleVertexTweakZ_ : SingleVertexTweakZPlug = PlugDescriptor("singleVertexTweakZ")
	stwz_ : SingleVertexTweakZPlug = PlugDescriptor("singleVertexTweakZ")
	node : Subdiv = None
	pass
class VertexTweakPlug(Plug):
	singleVertexTweak_ : SingleVertexTweakPlug = PlugDescriptor("singleVertexTweak")
	stw_ : SingleVertexTweakPlug = PlugDescriptor("singleVertexTweak")
	node : Subdiv = None
	pass
class WorldSubdivPlug(Plug):
	node : Subdiv = None
	pass
# endregion


# define node class
class Subdiv(SurfaceShape):
	baseFaceCount_ : BaseFaceCountPlug = PlugDescriptor("baseFaceCount")
	cached_ : CachedPlug = PlugDescriptor("cached")
	create_ : CreatePlug = PlugDescriptor("create")
	depth_ : DepthPlug = PlugDescriptor("depth")
	dispCreases_ : DispCreasesPlug = PlugDescriptor("dispCreases")
	dispEdges_ : DispEdgesPlug = PlugDescriptor("dispEdges")
	dispFaces_ : DispFacesPlug = PlugDescriptor("dispFaces")
	dispGeometry_ : DispGeometryPlug = PlugDescriptor("dispGeometry")
	dispMaps_ : DispMapsPlug = PlugDescriptor("dispMaps")
	dispResolution_ : DispResolutionPlug = PlugDescriptor("dispResolution")
	dispUVBorder_ : DispUVBorderPlug = PlugDescriptor("dispUVBorder")
	dispVertices_ : DispVerticesPlug = PlugDescriptor("dispVertices")
	dispVerticesAsLimitPoints_ : DispVerticesAsLimitPointsPlug = PlugDescriptor("dispVerticesAsLimitPoints")
	displayFilter_ : DisplayFilterPlug = PlugDescriptor("displayFilter")
	displayLevel_ : DisplayLevelPlug = PlugDescriptor("displayLevel")
	edgeCrease_ : EdgeCreasePlug = PlugDescriptor("edgeCrease")
	faceUVIds_ : FaceUVIdsPlug = PlugDescriptor("faceUVIds")
	format_ : FormatPlug = PlugDescriptor("format")
	levelOneFaceCount_ : LevelOneFaceCountPlug = PlugDescriptor("levelOneFaceCount")
	localizeLimitPointsEdit_ : LocalizeLimitPointsEditPlug = PlugDescriptor("localizeLimitPointsEdit")
	normalsDisplayScale_ : NormalsDisplayScalePlug = PlugDescriptor("normalsDisplayScale")
	outSubdiv_ : OutSubdivPlug = PlugDescriptor("outSubdiv")
	sampleCount_ : SampleCountPlug = PlugDescriptor("sampleCount")
	scalingHierarchy_ : ScalingHierarchyPlug = PlugDescriptor("scalingHierarchy")
	textureCoord_ : TextureCoordPlug = PlugDescriptor("textureCoord")
	singleVertexX_ : SingleVertexXPlug = PlugDescriptor("singleVertexX")
	singleVertexY_ : SingleVertexYPlug = PlugDescriptor("singleVertexY")
	singleVertexZ_ : SingleVertexZPlug = PlugDescriptor("singleVertexZ")
	singleVertex_ : SingleVertexPlug = PlugDescriptor("singleVertex")
	vertex_ : VertexPlug = PlugDescriptor("vertex")
	singleVertexTweakX_ : SingleVertexTweakXPlug = PlugDescriptor("singleVertexTweakX")
	singleVertexTweakY_ : SingleVertexTweakYPlug = PlugDescriptor("singleVertexTweakY")
	singleVertexTweakZ_ : SingleVertexTweakZPlug = PlugDescriptor("singleVertexTweakZ")
	singleVertexTweak_ : SingleVertexTweakPlug = PlugDescriptor("singleVertexTweak")
	vertexTweak_ : VertexTweakPlug = PlugDescriptor("vertexTweak")
	worldSubdiv_ : WorldSubdivPlug = PlugDescriptor("worldSubdiv")

	# node attributes

	typeName = "subdiv"
	apiTypeInt = 684
	apiTypeStr = "kSubdiv"
	typeIdInt = 1396986707
	MFnCls = om.MFnDagNode
	pass

