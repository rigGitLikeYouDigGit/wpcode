

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	SurfaceShape = Catalogue.SurfaceShape
else:
	from .. import retriever
	SurfaceShape = retriever.getNodeCls("SurfaceShape")
	assert SurfaceShape

# add node doc



# region plug type defs
class AllowTopologyModPlug(Plug):
	node : Mesh = None
	pass
class AlwaysDrawOnTopPlug(Plug):
	node : Mesh = None
	pass
class BackfaceCullingPlug(Plug):
	node : Mesh = None
	pass
class BorderWidthPlug(Plug):
	node : Mesh = None
	pass
class BoundaryRulePlug(Plug):
	node : Mesh = None
	pass
class CachedInMeshPlug(Plug):
	node : Mesh = None
	pass
class CachedSmoothMeshPlug(Plug):
	node : Mesh = None
	pass
class VertexColorRGBPlug(Plug):
	parent : VertexColorPlug = PlugDescriptor("vertexColor")
	vertexColorB_ : VertexColorBPlug = PlugDescriptor("vertexColorB")
	vxcb_ : VertexColorBPlug = PlugDescriptor("vertexColorB")
	vertexColorG_ : VertexColorGPlug = PlugDescriptor("vertexColorG")
	vxcg_ : VertexColorGPlug = PlugDescriptor("vertexColorG")
	vertexColorR_ : VertexColorRPlug = PlugDescriptor("vertexColorR")
	vxcr_ : VertexColorRPlug = PlugDescriptor("vertexColorR")
	node : Mesh = None
	pass
class VertexFaceColorRGBPlug(Plug):
	parent : VertexFaceColorPlug = PlugDescriptor("vertexFaceColor")
	vertexFaceColorB_ : VertexFaceColorBPlug = PlugDescriptor("vertexFaceColorB")
	vfcb_ : VertexFaceColorBPlug = PlugDescriptor("vertexFaceColorB")
	vertexFaceColorG_ : VertexFaceColorGPlug = PlugDescriptor("vertexFaceColorG")
	vfcg_ : VertexFaceColorGPlug = PlugDescriptor("vertexFaceColorG")
	vertexFaceColorR_ : VertexFaceColorRPlug = PlugDescriptor("vertexFaceColorR")
	vfcr_ : VertexFaceColorRPlug = PlugDescriptor("vertexFaceColorR")
	node : Mesh = None
	pass
class VertexFaceColorPlug(Plug):
	parent : VertexColorPlug = PlugDescriptor("vertexColor")
	vertexFaceAlpha_ : VertexFaceAlphaPlug = PlugDescriptor("vertexFaceAlpha")
	vfal_ : VertexFaceAlphaPlug = PlugDescriptor("vertexFaceAlpha")
	vertexFaceColorRGB_ : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	frgb_ : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	node : Mesh = None
	pass
class VertexColorPlug(Plug):
	parent : ColorPerVertexPlug = PlugDescriptor("colorPerVertex")
	vertexAlpha_ : VertexAlphaPlug = PlugDescriptor("vertexAlpha")
	vxal_ : VertexAlphaPlug = PlugDescriptor("vertexAlpha")
	vertexColorRGB_ : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	vrgb_ : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	vertexFaceColor_ : VertexFaceColorPlug = PlugDescriptor("vertexFaceColor")
	vfcl_ : VertexFaceColorPlug = PlugDescriptor("vertexFaceColor")
	node : Mesh = None
	pass
class ColorPerVertexPlug(Plug):
	vertexColor_ : VertexColorPlug = PlugDescriptor("vertexColor")
	vclr_ : VertexColorPlug = PlugDescriptor("vertexColor")
	node : Mesh = None
	pass
class ColorAPlug(Plug):
	parent : ColorsPlug = PlugDescriptor("colors")
	node : Mesh = None
	pass
class ColorBPlug(Plug):
	parent : ColorsPlug = PlugDescriptor("colors")
	node : Mesh = None
	pass
class ColorGPlug(Plug):
	parent : ColorsPlug = PlugDescriptor("colors")
	node : Mesh = None
	pass
class ColorRPlug(Plug):
	parent : ColorsPlug = PlugDescriptor("colors")
	node : Mesh = None
	pass
class ColorsPlug(Plug):
	colorA_ : ColorAPlug = PlugDescriptor("colorA")
	clra_ : ColorAPlug = PlugDescriptor("colorA")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	clrb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	clrg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	clrr_ : ColorRPlug = PlugDescriptor("colorR")
	node : Mesh = None
	pass
class ComputeFromSculptCachePlug(Plug):
	node : Mesh = None
	pass
class ContinuityPlug(Plug):
	node : Mesh = None
	pass
class CreaseDataPlug(Plug):
	node : Mesh = None
	pass
class CreaseVertexDataPlug(Plug):
	node : Mesh = None
	pass
class DispResolutionPlug(Plug):
	node : Mesh = None
	pass
class DisplacementTypePlug(Plug):
	node : Mesh = None
	pass
class DisplayAlphaAsGreyScalePlug(Plug):
	node : Mesh = None
	pass
class DisplayBlueColorChannelPlug(Plug):
	node : Mesh = None
	pass
class DisplayBordersPlug(Plug):
	node : Mesh = None
	pass
class DisplayCenterPlug(Plug):
	node : Mesh = None
	pass
class DisplayColorAsGreyScalePlug(Plug):
	node : Mesh = None
	pass
class DisplayEdgesPlug(Plug):
	node : Mesh = None
	pass
class DisplayFacesWithGroupIdPlug(Plug):
	node : Mesh = None
	pass
class DisplayGreenColorChannelPlug(Plug):
	node : Mesh = None
	pass
class DisplayInvisibleFacesPlug(Plug):
	node : Mesh = None
	pass
class DisplayItemNumbersPlug(Plug):
	node : Mesh = None
	pass
class DisplayMapBordersPlug(Plug):
	node : Mesh = None
	pass
class DisplayNonPlanarPlug(Plug):
	node : Mesh = None
	pass
class DisplayNormalPlug(Plug):
	node : Mesh = None
	pass
class DisplayRedColorChannelPlug(Plug):
	node : Mesh = None
	pass
class DisplaySmoothMeshPlug(Plug):
	node : Mesh = None
	pass
class DisplaySubdCompsPlug(Plug):
	node : Mesh = None
	pass
class DisplayTangentPlug(Plug):
	node : Mesh = None
	pass
class DisplayTrianglesPlug(Plug):
	node : Mesh = None
	pass
class DisplayUVsPlug(Plug):
	node : Mesh = None
	pass
class DisplayVerticesPlug(Plug):
	node : Mesh = None
	pass
class Edg1Plug(Plug):
	parent : EdgePlug = PlugDescriptor("edge")
	node : Mesh = None
	pass
class Edg2Plug(Plug):
	parent : EdgePlug = PlugDescriptor("edge")
	node : Mesh = None
	pass
class EdghPlug(Plug):
	parent : EdgePlug = PlugDescriptor("edge")
	node : Mesh = None
	pass
class EdgePlug(Plug):
	edg1_ : Edg1Plug = PlugDescriptor("edg1")
	e1_ : Edg1Plug = PlugDescriptor("edg1")
	edg2_ : Edg2Plug = PlugDescriptor("edg2")
	e2_ : Edg2Plug = PlugDescriptor("edg2")
	edgh_ : EdghPlug = PlugDescriptor("edgh")
	eh_ : EdghPlug = PlugDescriptor("edgh")
	node : Mesh = None
	pass
class EdgeIdMapPlug(Plug):
	node : Mesh = None
	pass
class EnableOpenCLPlug(Plug):
	node : Mesh = None
	pass
class FacePlug(Plug):
	node : Mesh = None
	pass
class FaceColorIndicesPlug(Plug):
	node : Mesh = None
	pass
class FaceIdMapPlug(Plug):
	node : Mesh = None
	pass
class FreezePlug(Plug):
	node : Mesh = None
	pass
class HoleFaceDataPlug(Plug):
	node : Mesh = None
	pass
class InForceNodeUVUpdatePlug(Plug):
	node : Mesh = None
	pass
class InMeshPlug(Plug):
	node : Mesh = None
	pass
class KeepBorderPlug(Plug):
	node : Mesh = None
	pass
class KeepHardEdgePlug(Plug):
	node : Mesh = None
	pass
class KeepMapBordersPlug(Plug):
	node : Mesh = None
	pass
class LoadTiledTexturesPlug(Plug):
	node : Mesh = None
	pass
class MaterialBlendPlug(Plug):
	node : Mesh = None
	pass
class MaxEdgeLengthPlug(Plug):
	node : Mesh = None
	pass
class MaxSubdPlug(Plug):
	node : Mesh = None
	pass
class MaxTrianglesPlug(Plug):
	node : Mesh = None
	pass
class MaxUvPlug(Plug):
	node : Mesh = None
	pass
class MinEdgeLengthPlug(Plug):
	node : Mesh = None
	pass
class MinScreenPlug(Plug):
	node : Mesh = None
	pass
class MotionVectorColorSetPlug(Plug):
	node : Mesh = None
	pass
class VertexNormalYPlug(Plug):
	parent : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	node : Mesh = None
	pass
class VertexNormalZPlug(Plug):
	parent : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	node : Mesh = None
	pass
class VertexNormalXYZPlug(Plug):
	parent : VertexNormalPlug = PlugDescriptor("vertexNormal")
	vertexNormalX_ : VertexNormalXPlug = PlugDescriptor("vertexNormalX")
	vxnx_ : VertexNormalXPlug = PlugDescriptor("vertexNormalX")
	vertexNormalY_ : VertexNormalYPlug = PlugDescriptor("vertexNormalY")
	vxny_ : VertexNormalYPlug = PlugDescriptor("vertexNormalY")
	vertexNormalZ_ : VertexNormalZPlug = PlugDescriptor("vertexNormalZ")
	vxnz_ : VertexNormalZPlug = PlugDescriptor("vertexNormalZ")
	node : Mesh = None
	pass
class VertexNormalPlug(Plug):
	parent : NormalPerVertexPlug = PlugDescriptor("normalPerVertex")
	vertexFaceNormal_ : VertexFaceNormalPlug = PlugDescriptor("vertexFaceNormal")
	vfnl_ : VertexFaceNormalPlug = PlugDescriptor("vertexFaceNormal")
	vertexNormalXYZ_ : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	nxyz_ : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	node : Mesh = None
	pass
class NormalPerVertexPlug(Plug):
	vertexNormal_ : VertexNormalPlug = PlugDescriptor("vertexNormal")
	vn_ : VertexNormalPlug = PlugDescriptor("vertexNormal")
	node : Mesh = None
	pass
class NormalSizePlug(Plug):
	node : Mesh = None
	pass
class NormalTypePlug(Plug):
	node : Mesh = None
	pass
class NormalxPlug(Plug):
	parent : NormalsPlug = PlugDescriptor("normals")
	node : Mesh = None
	pass
class NormalyPlug(Plug):
	parent : NormalsPlug = PlugDescriptor("normals")
	node : Mesh = None
	pass
class NormalzPlug(Plug):
	parent : NormalsPlug = PlugDescriptor("normals")
	node : Mesh = None
	pass
class NormalsPlug(Plug):
	normalx_ : NormalxPlug = PlugDescriptor("normalx")
	nx_ : NormalxPlug = PlugDescriptor("normalx")
	normaly_ : NormalyPlug = PlugDescriptor("normaly")
	ny_ : NormalyPlug = PlugDescriptor("normaly")
	normalz_ : NormalzPlug = PlugDescriptor("normalz")
	nz_ : NormalzPlug = PlugDescriptor("normalz")
	node : Mesh = None
	pass
class NumTrianglesPlug(Plug):
	node : Mesh = None
	pass
class OsdCreaseMethodPlug(Plug):
	node : Mesh = None
	pass
class OsdFvarBoundaryPlug(Plug):
	node : Mesh = None
	pass
class OsdFvarPropagateCornersPlug(Plug):
	node : Mesh = None
	pass
class OsdIndependentUVChannelsPlug(Plug):
	node : Mesh = None
	pass
class OsdSmoothTrianglesPlug(Plug):
	node : Mesh = None
	pass
class OsdVertBoundaryPlug(Plug):
	node : Mesh = None
	pass
class OutForceNodeUVUpdatePlug(Plug):
	node : Mesh = None
	pass
class OutGeometryCleanPlug(Plug):
	node : Mesh = None
	pass
class OutMeshPlug(Plug):
	node : Mesh = None
	pass
class OutSmoothMeshPlug(Plug):
	node : Mesh = None
	pass
class OutSmoothMeshSubdErrorPlug(Plug):
	node : Mesh = None
	pass
class PerInstanceIndexPlug(Plug):
	node : Mesh = None
	pass
class PerInstanceTagPlug(Plug):
	node : Mesh = None
	pass
class PinDataPlug(Plug):
	node : Mesh = None
	pass
class PntxPlug(Plug):
	parent : PntsPlug = PlugDescriptor("pnts")
	node : Mesh = None
	pass
class PntyPlug(Plug):
	parent : PntsPlug = PlugDescriptor("pnts")
	node : Mesh = None
	pass
class PntzPlug(Plug):
	parent : PntsPlug = PlugDescriptor("pnts")
	node : Mesh = None
	pass
class PntsPlug(Plug):
	pntx_ : PntxPlug = PlugDescriptor("pntx")
	px_ : PntxPlug = PlugDescriptor("pntx")
	pnty_ : PntyPlug = PlugDescriptor("pnty")
	py_ : PntyPlug = PlugDescriptor("pnty")
	pntz_ : PntzPlug = PlugDescriptor("pntz")
	pz_ : PntzPlug = PlugDescriptor("pntz")
	node : Mesh = None
	pass
class PropagateEdgeHardnessPlug(Plug):
	node : Mesh = None
	pass
class QuadSplitPlug(Plug):
	node : Mesh = None
	pass
class RenderSmoothLevelPlug(Plug):
	node : Mesh = None
	pass
class ReuseTrianglesPlug(Plug):
	node : Mesh = None
	pass
class ShowDisplacementsPlug(Plug):
	node : Mesh = None
	pass
class SmoothDrawTypePlug(Plug):
	node : Mesh = None
	pass
class SmoothLevelPlug(Plug):
	node : Mesh = None
	pass
class SmoothMeshSelectionModePlug(Plug):
	node : Mesh = None
	pass
class SofxPlug(Plug):
	parent : SmoothOffsetPlug = PlugDescriptor("smoothOffset")
	node : Mesh = None
	pass
class SofyPlug(Plug):
	parent : SmoothOffsetPlug = PlugDescriptor("smoothOffset")
	node : Mesh = None
	pass
class SofzPlug(Plug):
	parent : SmoothOffsetPlug = PlugDescriptor("smoothOffset")
	node : Mesh = None
	pass
class SmoothOffsetPlug(Plug):
	sofx_ : SofxPlug = PlugDescriptor("sofx")
	sx_ : SofxPlug = PlugDescriptor("sofx")
	sofy_ : SofyPlug = PlugDescriptor("sofy")
	sy_ : SofyPlug = PlugDescriptor("sofy")
	sofz_ : SofzPlug = PlugDescriptor("sofz")
	sz_ : SofzPlug = PlugDescriptor("sofz")
	node : Mesh = None
	pass
class SmoothOsdColorizePatchesPlug(Plug):
	node : Mesh = None
	pass
class SmoothTessLevelPlug(Plug):
	node : Mesh = None
	pass
class SmoothUVsPlug(Plug):
	node : Mesh = None
	pass
class SmoothWarnPlug(Plug):
	node : Mesh = None
	pass
class TangentNormalThresholdPlug(Plug):
	node : Mesh = None
	pass
class TangentSmoothingAnglePlug(Plug):
	node : Mesh = None
	pass
class TangentSpacePlug(Plug):
	node : Mesh = None
	pass
class UseGlobalSmoothDrawTypePlug(Plug):
	node : Mesh = None
	pass
class UseMaxEdgeLengthPlug(Plug):
	node : Mesh = None
	pass
class UseMaxSubdivisionsPlug(Plug):
	node : Mesh = None
	pass
class UseMaxUVPlug(Plug):
	node : Mesh = None
	pass
class UseMeshSculptCachePlug(Plug):
	node : Mesh = None
	pass
class UseMeshTexSculptCachePlug(Plug):
	node : Mesh = None
	pass
class UseMinEdgeLengthPlug(Plug):
	node : Mesh = None
	pass
class UseMinScreenPlug(Plug):
	node : Mesh = None
	pass
class UseNumTrianglesPlug(Plug):
	node : Mesh = None
	pass
class UseOsdBoundaryMethodsPlug(Plug):
	node : Mesh = None
	pass
class UseSmoothPreviewForRenderPlug(Plug):
	node : Mesh = None
	pass
class UserTrgPlug(Plug):
	node : Mesh = None
	pass
class UvSizePlug(Plug):
	node : Mesh = None
	pass
class UvTweakLocationPlug(Plug):
	node : Mesh = None
	pass
class UvpxPlug(Plug):
	parent : UvptPlug = PlugDescriptor("uvpt")
	node : Mesh = None
	pass
class UvpyPlug(Plug):
	parent : UvptPlug = PlugDescriptor("uvpt")
	node : Mesh = None
	pass
class UvptPlug(Plug):
	uvpx_ : UvpxPlug = PlugDescriptor("uvpx")
	ux_ : UvpxPlug = PlugDescriptor("uvpx")
	uvpy_ : UvpyPlug = PlugDescriptor("uvpy")
	uy_ : UvpyPlug = PlugDescriptor("uvpy")
	node : Mesh = None
	pass
class VertexBackfaceCullingPlug(Plug):
	node : Mesh = None
	pass
class VertexAlphaPlug(Plug):
	parent : VertexColorPlug = PlugDescriptor("vertexColor")
	node : Mesh = None
	pass
class VertexColorBPlug(Plug):
	parent : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	node : Mesh = None
	pass
class VertexColorGPlug(Plug):
	parent : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	node : Mesh = None
	pass
class VertexColorRPlug(Plug):
	parent : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	node : Mesh = None
	pass
class VertexColorSourcePlug(Plug):
	node : Mesh = None
	pass
class VertexFaceAlphaPlug(Plug):
	parent : VertexFaceColorPlug = PlugDescriptor("vertexFaceColor")
	node : Mesh = None
	pass
class VertexFaceColorBPlug(Plug):
	parent : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	node : Mesh = None
	pass
class VertexFaceColorGPlug(Plug):
	parent : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	node : Mesh = None
	pass
class VertexFaceColorRPlug(Plug):
	parent : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	node : Mesh = None
	pass
class VertexFaceNormalXPlug(Plug):
	parent : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	node : Mesh = None
	pass
class VertexIdMapPlug(Plug):
	node : Mesh = None
	pass
class VertexFaceNormalYPlug(Plug):
	parent : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	node : Mesh = None
	pass
class VertexFaceNormalZPlug(Plug):
	parent : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	node : Mesh = None
	pass
class VertexFaceNormalXYZPlug(Plug):
	parent : VertexFaceNormalPlug = PlugDescriptor("vertexFaceNormal")
	vertexFaceNormalX_ : VertexFaceNormalXPlug = PlugDescriptor("vertexFaceNormalX")
	vfnx_ : VertexFaceNormalXPlug = PlugDescriptor("vertexFaceNormalX")
	vertexFaceNormalY_ : VertexFaceNormalYPlug = PlugDescriptor("vertexFaceNormalY")
	vfny_ : VertexFaceNormalYPlug = PlugDescriptor("vertexFaceNormalY")
	vertexFaceNormalZ_ : VertexFaceNormalZPlug = PlugDescriptor("vertexFaceNormalZ")
	vfnz_ : VertexFaceNormalZPlug = PlugDescriptor("vertexFaceNormalZ")
	node : Mesh = None
	pass
class VertexFaceNormalPlug(Plug):
	parent : VertexNormalPlug = PlugDescriptor("vertexNormal")
	vertexFaceNormalXYZ_ : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	fnxy_ : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	node : Mesh = None
	pass
class VertexNormalMethodPlug(Plug):
	node : Mesh = None
	pass
class VertexNormalXPlug(Plug):
	parent : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	node : Mesh = None
	pass
class VertexSizePlug(Plug):
	node : Mesh = None
	pass
class VrtxPlug(Plug):
	parent : VrtsPlug = PlugDescriptor("vrts")
	node : Mesh = None
	pass
class VrtyPlug(Plug):
	parent : VrtsPlug = PlugDescriptor("vrts")
	node : Mesh = None
	pass
class VrtzPlug(Plug):
	parent : VrtsPlug = PlugDescriptor("vrts")
	node : Mesh = None
	pass
class VrtsPlug(Plug):
	vrtx_ : VrtxPlug = PlugDescriptor("vrtx")
	vx_ : VrtxPlug = PlugDescriptor("vrtx")
	vrty_ : VrtyPlug = PlugDescriptor("vrty")
	vy_ : VrtyPlug = PlugDescriptor("vrty")
	vrtz_ : VrtzPlug = PlugDescriptor("vrtz")
	vz_ : VrtzPlug = PlugDescriptor("vrtz")
	node : Mesh = None
	pass
class WorldMeshPlug(Plug):
	node : Mesh = None
	pass
# endregion


# define node class
class Mesh(SurfaceShape):
	allowTopologyMod_ : AllowTopologyModPlug = PlugDescriptor("allowTopologyMod")
	alwaysDrawOnTop_ : AlwaysDrawOnTopPlug = PlugDescriptor("alwaysDrawOnTop")
	backfaceCulling_ : BackfaceCullingPlug = PlugDescriptor("backfaceCulling")
	borderWidth_ : BorderWidthPlug = PlugDescriptor("borderWidth")
	boundaryRule_ : BoundaryRulePlug = PlugDescriptor("boundaryRule")
	cachedInMesh_ : CachedInMeshPlug = PlugDescriptor("cachedInMesh")
	cachedSmoothMesh_ : CachedSmoothMeshPlug = PlugDescriptor("cachedSmoothMesh")
	vertexColorRGB_ : VertexColorRGBPlug = PlugDescriptor("vertexColorRGB")
	vertexFaceColorRGB_ : VertexFaceColorRGBPlug = PlugDescriptor("vertexFaceColorRGB")
	vertexFaceColor_ : VertexFaceColorPlug = PlugDescriptor("vertexFaceColor")
	vertexColor_ : VertexColorPlug = PlugDescriptor("vertexColor")
	colorPerVertex_ : ColorPerVertexPlug = PlugDescriptor("colorPerVertex")
	colorA_ : ColorAPlug = PlugDescriptor("colorA")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	colors_ : ColorsPlug = PlugDescriptor("colors")
	computeFromSculptCache_ : ComputeFromSculptCachePlug = PlugDescriptor("computeFromSculptCache")
	continuity_ : ContinuityPlug = PlugDescriptor("continuity")
	creaseData_ : CreaseDataPlug = PlugDescriptor("creaseData")
	creaseVertexData_ : CreaseVertexDataPlug = PlugDescriptor("creaseVertexData")
	dispResolution_ : DispResolutionPlug = PlugDescriptor("dispResolution")
	displacementType_ : DisplacementTypePlug = PlugDescriptor("displacementType")
	displayAlphaAsGreyScale_ : DisplayAlphaAsGreyScalePlug = PlugDescriptor("displayAlphaAsGreyScale")
	displayBlueColorChannel_ : DisplayBlueColorChannelPlug = PlugDescriptor("displayBlueColorChannel")
	displayBorders_ : DisplayBordersPlug = PlugDescriptor("displayBorders")
	displayCenter_ : DisplayCenterPlug = PlugDescriptor("displayCenter")
	displayColorAsGreyScale_ : DisplayColorAsGreyScalePlug = PlugDescriptor("displayColorAsGreyScale")
	displayEdges_ : DisplayEdgesPlug = PlugDescriptor("displayEdges")
	displayFacesWithGroupId_ : DisplayFacesWithGroupIdPlug = PlugDescriptor("displayFacesWithGroupId")
	displayGreenColorChannel_ : DisplayGreenColorChannelPlug = PlugDescriptor("displayGreenColorChannel")
	displayInvisibleFaces_ : DisplayInvisibleFacesPlug = PlugDescriptor("displayInvisibleFaces")
	displayItemNumbers_ : DisplayItemNumbersPlug = PlugDescriptor("displayItemNumbers")
	displayMapBorders_ : DisplayMapBordersPlug = PlugDescriptor("displayMapBorders")
	displayNonPlanar_ : DisplayNonPlanarPlug = PlugDescriptor("displayNonPlanar")
	displayNormal_ : DisplayNormalPlug = PlugDescriptor("displayNormal")
	displayRedColorChannel_ : DisplayRedColorChannelPlug = PlugDescriptor("displayRedColorChannel")
	displaySmoothMesh_ : DisplaySmoothMeshPlug = PlugDescriptor("displaySmoothMesh")
	displaySubdComps_ : DisplaySubdCompsPlug = PlugDescriptor("displaySubdComps")
	displayTangent_ : DisplayTangentPlug = PlugDescriptor("displayTangent")
	displayTriangles_ : DisplayTrianglesPlug = PlugDescriptor("displayTriangles")
	displayUVs_ : DisplayUVsPlug = PlugDescriptor("displayUVs")
	displayVertices_ : DisplayVerticesPlug = PlugDescriptor("displayVertices")
	edg1_ : Edg1Plug = PlugDescriptor("edg1")
	edg2_ : Edg2Plug = PlugDescriptor("edg2")
	edgh_ : EdghPlug = PlugDescriptor("edgh")
	edge_ : EdgePlug = PlugDescriptor("edge")
	edgeIdMap_ : EdgeIdMapPlug = PlugDescriptor("edgeIdMap")
	enableOpenCL_ : EnableOpenCLPlug = PlugDescriptor("enableOpenCL")
	face_ : FacePlug = PlugDescriptor("face")
	faceColorIndices_ : FaceColorIndicesPlug = PlugDescriptor("faceColorIndices")
	faceIdMap_ : FaceIdMapPlug = PlugDescriptor("faceIdMap")
	freeze_ : FreezePlug = PlugDescriptor("freeze")
	holeFaceData_ : HoleFaceDataPlug = PlugDescriptor("holeFaceData")
	inForceNodeUVUpdate_ : InForceNodeUVUpdatePlug = PlugDescriptor("inForceNodeUVUpdate")
	inMesh_ : InMeshPlug = PlugDescriptor("inMesh")
	keepBorder_ : KeepBorderPlug = PlugDescriptor("keepBorder")
	keepHardEdge_ : KeepHardEdgePlug = PlugDescriptor("keepHardEdge")
	keepMapBorders_ : KeepMapBordersPlug = PlugDescriptor("keepMapBorders")
	loadTiledTextures_ : LoadTiledTexturesPlug = PlugDescriptor("loadTiledTextures")
	materialBlend_ : MaterialBlendPlug = PlugDescriptor("materialBlend")
	maxEdgeLength_ : MaxEdgeLengthPlug = PlugDescriptor("maxEdgeLength")
	maxSubd_ : MaxSubdPlug = PlugDescriptor("maxSubd")
	maxTriangles_ : MaxTrianglesPlug = PlugDescriptor("maxTriangles")
	maxUv_ : MaxUvPlug = PlugDescriptor("maxUv")
	minEdgeLength_ : MinEdgeLengthPlug = PlugDescriptor("minEdgeLength")
	minScreen_ : MinScreenPlug = PlugDescriptor("minScreen")
	motionVectorColorSet_ : MotionVectorColorSetPlug = PlugDescriptor("motionVectorColorSet")
	vertexNormalY_ : VertexNormalYPlug = PlugDescriptor("vertexNormalY")
	vertexNormalZ_ : VertexNormalZPlug = PlugDescriptor("vertexNormalZ")
	vertexNormalXYZ_ : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	vertexNormal_ : VertexNormalPlug = PlugDescriptor("vertexNormal")
	normalPerVertex_ : NormalPerVertexPlug = PlugDescriptor("normalPerVertex")
	normalSize_ : NormalSizePlug = PlugDescriptor("normalSize")
	normalType_ : NormalTypePlug = PlugDescriptor("normalType")
	normalx_ : NormalxPlug = PlugDescriptor("normalx")
	normaly_ : NormalyPlug = PlugDescriptor("normaly")
	normalz_ : NormalzPlug = PlugDescriptor("normalz")
	normals_ : NormalsPlug = PlugDescriptor("normals")
	numTriangles_ : NumTrianglesPlug = PlugDescriptor("numTriangles")
	osdCreaseMethod_ : OsdCreaseMethodPlug = PlugDescriptor("osdCreaseMethod")
	osdFvarBoundary_ : OsdFvarBoundaryPlug = PlugDescriptor("osdFvarBoundary")
	osdFvarPropagateCorners_ : OsdFvarPropagateCornersPlug = PlugDescriptor("osdFvarPropagateCorners")
	osdIndependentUVChannels_ : OsdIndependentUVChannelsPlug = PlugDescriptor("osdIndependentUVChannels")
	osdSmoothTriangles_ : OsdSmoothTrianglesPlug = PlugDescriptor("osdSmoothTriangles")
	osdVertBoundary_ : OsdVertBoundaryPlug = PlugDescriptor("osdVertBoundary")
	outForceNodeUVUpdate_ : OutForceNodeUVUpdatePlug = PlugDescriptor("outForceNodeUVUpdate")
	outGeometryClean_ : OutGeometryCleanPlug = PlugDescriptor("outGeometryClean")
	outMesh_ : OutMeshPlug = PlugDescriptor("outMesh")
	outSmoothMesh_ : OutSmoothMeshPlug = PlugDescriptor("outSmoothMesh")
	outSmoothMeshSubdError_ : OutSmoothMeshSubdErrorPlug = PlugDescriptor("outSmoothMeshSubdError")
	perInstanceIndex_ : PerInstanceIndexPlug = PlugDescriptor("perInstanceIndex")
	perInstanceTag_ : PerInstanceTagPlug = PlugDescriptor("perInstanceTag")
	pinData_ : PinDataPlug = PlugDescriptor("pinData")
	pntx_ : PntxPlug = PlugDescriptor("pntx")
	pnty_ : PntyPlug = PlugDescriptor("pnty")
	pntz_ : PntzPlug = PlugDescriptor("pntz")
	pnts_ : PntsPlug = PlugDescriptor("pnts")
	propagateEdgeHardness_ : PropagateEdgeHardnessPlug = PlugDescriptor("propagateEdgeHardness")
	quadSplit_ : QuadSplitPlug = PlugDescriptor("quadSplit")
	renderSmoothLevel_ : RenderSmoothLevelPlug = PlugDescriptor("renderSmoothLevel")
	reuseTriangles_ : ReuseTrianglesPlug = PlugDescriptor("reuseTriangles")
	showDisplacements_ : ShowDisplacementsPlug = PlugDescriptor("showDisplacements")
	smoothDrawType_ : SmoothDrawTypePlug = PlugDescriptor("smoothDrawType")
	smoothLevel_ : SmoothLevelPlug = PlugDescriptor("smoothLevel")
	smoothMeshSelectionMode_ : SmoothMeshSelectionModePlug = PlugDescriptor("smoothMeshSelectionMode")
	sofx_ : SofxPlug = PlugDescriptor("sofx")
	sofy_ : SofyPlug = PlugDescriptor("sofy")
	sofz_ : SofzPlug = PlugDescriptor("sofz")
	smoothOffset_ : SmoothOffsetPlug = PlugDescriptor("smoothOffset")
	smoothOsdColorizePatches_ : SmoothOsdColorizePatchesPlug = PlugDescriptor("smoothOsdColorizePatches")
	smoothTessLevel_ : SmoothTessLevelPlug = PlugDescriptor("smoothTessLevel")
	smoothUVs_ : SmoothUVsPlug = PlugDescriptor("smoothUVs")
	smoothWarn_ : SmoothWarnPlug = PlugDescriptor("smoothWarn")
	tangentNormalThreshold_ : TangentNormalThresholdPlug = PlugDescriptor("tangentNormalThreshold")
	tangentSmoothingAngle_ : TangentSmoothingAnglePlug = PlugDescriptor("tangentSmoothingAngle")
	tangentSpace_ : TangentSpacePlug = PlugDescriptor("tangentSpace")
	useGlobalSmoothDrawType_ : UseGlobalSmoothDrawTypePlug = PlugDescriptor("useGlobalSmoothDrawType")
	useMaxEdgeLength_ : UseMaxEdgeLengthPlug = PlugDescriptor("useMaxEdgeLength")
	useMaxSubdivisions_ : UseMaxSubdivisionsPlug = PlugDescriptor("useMaxSubdivisions")
	useMaxUV_ : UseMaxUVPlug = PlugDescriptor("useMaxUV")
	useMeshSculptCache_ : UseMeshSculptCachePlug = PlugDescriptor("useMeshSculptCache")
	useMeshTexSculptCache_ : UseMeshTexSculptCachePlug = PlugDescriptor("useMeshTexSculptCache")
	useMinEdgeLength_ : UseMinEdgeLengthPlug = PlugDescriptor("useMinEdgeLength")
	useMinScreen_ : UseMinScreenPlug = PlugDescriptor("useMinScreen")
	useNumTriangles_ : UseNumTrianglesPlug = PlugDescriptor("useNumTriangles")
	useOsdBoundaryMethods_ : UseOsdBoundaryMethodsPlug = PlugDescriptor("useOsdBoundaryMethods")
	useSmoothPreviewForRender_ : UseSmoothPreviewForRenderPlug = PlugDescriptor("useSmoothPreviewForRender")
	userTrg_ : UserTrgPlug = PlugDescriptor("userTrg")
	uvSize_ : UvSizePlug = PlugDescriptor("uvSize")
	uvTweakLocation_ : UvTweakLocationPlug = PlugDescriptor("uvTweakLocation")
	uvpx_ : UvpxPlug = PlugDescriptor("uvpx")
	uvpy_ : UvpyPlug = PlugDescriptor("uvpy")
	uvpt_ : UvptPlug = PlugDescriptor("uvpt")
	vertexBackfaceCulling_ : VertexBackfaceCullingPlug = PlugDescriptor("vertexBackfaceCulling")
	vertexAlpha_ : VertexAlphaPlug = PlugDescriptor("vertexAlpha")
	vertexColorB_ : VertexColorBPlug = PlugDescriptor("vertexColorB")
	vertexColorG_ : VertexColorGPlug = PlugDescriptor("vertexColorG")
	vertexColorR_ : VertexColorRPlug = PlugDescriptor("vertexColorR")
	vertexColorSource_ : VertexColorSourcePlug = PlugDescriptor("vertexColorSource")
	vertexFaceAlpha_ : VertexFaceAlphaPlug = PlugDescriptor("vertexFaceAlpha")
	vertexFaceColorB_ : VertexFaceColorBPlug = PlugDescriptor("vertexFaceColorB")
	vertexFaceColorG_ : VertexFaceColorGPlug = PlugDescriptor("vertexFaceColorG")
	vertexFaceColorR_ : VertexFaceColorRPlug = PlugDescriptor("vertexFaceColorR")
	vertexFaceNormalX_ : VertexFaceNormalXPlug = PlugDescriptor("vertexFaceNormalX")
	vertexIdMap_ : VertexIdMapPlug = PlugDescriptor("vertexIdMap")
	vertexFaceNormalY_ : VertexFaceNormalYPlug = PlugDescriptor("vertexFaceNormalY")
	vertexFaceNormalZ_ : VertexFaceNormalZPlug = PlugDescriptor("vertexFaceNormalZ")
	vertexFaceNormalXYZ_ : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	vertexFaceNormal_ : VertexFaceNormalPlug = PlugDescriptor("vertexFaceNormal")
	vertexNormalMethod_ : VertexNormalMethodPlug = PlugDescriptor("vertexNormalMethod")
	vertexNormalX_ : VertexNormalXPlug = PlugDescriptor("vertexNormalX")
	vertexSize_ : VertexSizePlug = PlugDescriptor("vertexSize")
	vrtx_ : VrtxPlug = PlugDescriptor("vrtx")
	vrty_ : VrtyPlug = PlugDescriptor("vrty")
	vrtz_ : VrtzPlug = PlugDescriptor("vrtz")
	vrts_ : VrtsPlug = PlugDescriptor("vrts")
	worldMesh_ : WorldMeshPlug = PlugDescriptor("worldMesh")

	# node attributes

	typeName = "mesh"
	apiTypeInt = 296
	apiTypeStr = "kMesh"
	typeIdInt = 1145918280
	MFnCls = om.MFnMesh
	nodeLeafClassAttrs = ["allowTopologyMod", "alwaysDrawOnTop", "backfaceCulling", "borderWidth", "boundaryRule", "cachedInMesh", "cachedSmoothMesh", "vertexColorRGB", "vertexFaceColorRGB", "vertexFaceColor", "vertexColor", "colorPerVertex", "colorA", "colorB", "colorG", "colorR", "colors", "computeFromSculptCache", "continuity", "creaseData", "creaseVertexData", "dispResolution", "displacementType", "displayAlphaAsGreyScale", "displayBlueColorChannel", "displayBorders", "displayCenter", "displayColorAsGreyScale", "displayEdges", "displayFacesWithGroupId", "displayGreenColorChannel", "displayInvisibleFaces", "displayItemNumbers", "displayMapBorders", "displayNonPlanar", "displayNormal", "displayRedColorChannel", "displaySmoothMesh", "displaySubdComps", "displayTangent", "displayTriangles", "displayUVs", "displayVertices", "edg1", "edg2", "edgh", "edge", "edgeIdMap", "enableOpenCL", "face", "faceColorIndices", "faceIdMap", "freeze", "holeFaceData", "inForceNodeUVUpdate", "inMesh", "keepBorder", "keepHardEdge", "keepMapBorders", "loadTiledTextures", "materialBlend", "maxEdgeLength", "maxSubd", "maxTriangles", "maxUv", "minEdgeLength", "minScreen", "motionVectorColorSet", "vertexNormalY", "vertexNormalZ", "vertexNormalXYZ", "vertexNormal", "normalPerVertex", "normalSize", "normalType", "normalx", "normaly", "normalz", "normals", "numTriangles", "osdCreaseMethod", "osdFvarBoundary", "osdFvarPropagateCorners", "osdIndependentUVChannels", "osdSmoothTriangles", "osdVertBoundary", "outForceNodeUVUpdate", "outGeometryClean", "outMesh", "outSmoothMesh", "outSmoothMeshSubdError", "perInstanceIndex", "perInstanceTag", "pinData", "pntx", "pnty", "pntz", "pnts", "propagateEdgeHardness", "quadSplit", "renderSmoothLevel", "reuseTriangles", "showDisplacements", "smoothDrawType", "smoothLevel", "smoothMeshSelectionMode", "sofx", "sofy", "sofz", "smoothOffset", "smoothOsdColorizePatches", "smoothTessLevel", "smoothUVs", "smoothWarn", "tangentNormalThreshold", "tangentSmoothingAngle", "tangentSpace", "useGlobalSmoothDrawType", "useMaxEdgeLength", "useMaxSubdivisions", "useMaxUV", "useMeshSculptCache", "useMeshTexSculptCache", "useMinEdgeLength", "useMinScreen", "useNumTriangles", "useOsdBoundaryMethods", "useSmoothPreviewForRender", "userTrg", "uvSize", "uvTweakLocation", "uvpx", "uvpy", "uvpt", "vertexBackfaceCulling", "vertexAlpha", "vertexColorB", "vertexColorG", "vertexColorR", "vertexColorSource", "vertexFaceAlpha", "vertexFaceColorB", "vertexFaceColorG", "vertexFaceColorR", "vertexFaceNormalX", "vertexIdMap", "vertexFaceNormalY", "vertexFaceNormalZ", "vertexFaceNormalXYZ", "vertexFaceNormal", "vertexNormalMethod", "vertexNormalX", "vertexSize", "vrtx", "vrty", "vrtz", "vrts", "worldMesh"]
	nodeLeafPlugs = ["allowTopologyMod", "alwaysDrawOnTop", "backfaceCulling", "borderWidth", "boundaryRule", "cachedInMesh", "cachedSmoothMesh", "colorPerVertex", "colors", "computeFromSculptCache", "continuity", "creaseData", "creaseVertexData", "dispResolution", "displacementType", "displayAlphaAsGreyScale", "displayBlueColorChannel", "displayBorders", "displayCenter", "displayColorAsGreyScale", "displayEdges", "displayFacesWithGroupId", "displayGreenColorChannel", "displayInvisibleFaces", "displayItemNumbers", "displayMapBorders", "displayNonPlanar", "displayNormal", "displayRedColorChannel", "displaySmoothMesh", "displaySubdComps", "displayTangent", "displayTriangles", "displayUVs", "displayVertices", "edge", "edgeIdMap", "enableOpenCL", "face", "faceColorIndices", "faceIdMap", "freeze", "holeFaceData", "inForceNodeUVUpdate", "inMesh", "keepBorder", "keepHardEdge", "keepMapBorders", "loadTiledTextures", "materialBlend", "maxEdgeLength", "maxSubd", "maxTriangles", "maxUv", "minEdgeLength", "minScreen", "motionVectorColorSet", "normalPerVertex", "normalSize", "normalType", "normals", "numTriangles", "osdCreaseMethod", "osdFvarBoundary", "osdFvarPropagateCorners", "osdIndependentUVChannels", "osdSmoothTriangles", "osdVertBoundary", "outForceNodeUVUpdate", "outGeometryClean", "outMesh", "outSmoothMesh", "outSmoothMeshSubdError", "perInstanceIndex", "perInstanceTag", "pinData", "pnts", "propagateEdgeHardness", "quadSplit", "renderSmoothLevel", "reuseTriangles", "showDisplacements", "smoothDrawType", "smoothLevel", "smoothMeshSelectionMode", "smoothOffset", "smoothOsdColorizePatches", "smoothTessLevel", "smoothUVs", "smoothWarn", "tangentNormalThreshold", "tangentSmoothingAngle", "tangentSpace", "useGlobalSmoothDrawType", "useMaxEdgeLength", "useMaxSubdivisions", "useMaxUV", "useMeshSculptCache", "useMeshTexSculptCache", "useMinEdgeLength", "useMinScreen", "useNumTriangles", "useOsdBoundaryMethods", "useSmoothPreviewForRender", "userTrg", "uvSize", "uvTweakLocation", "uvpt", "vertexBackfaceCulling", "vertexColorSource", "vertexIdMap", "vertexNormalMethod", "vertexSize", "vrts", "worldMesh"]
	pass

