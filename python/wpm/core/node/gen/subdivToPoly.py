

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class ApplyMatrixToResultPlug(Plug):
	node : SubdivToPoly = None
	pass
class BinMembershipPlug(Plug):
	node : SubdivToPoly = None
	pass
class ConvertCompPlug(Plug):
	node : SubdivToPoly = None
	pass
class CopyUVTopologyPlug(Plug):
	node : SubdivToPoly = None
	pass
class DepthPlug(Plug):
	node : SubdivToPoly = None
	pass
class ExtractPointPositionPlug(Plug):
	node : SubdivToPoly = None
	pass
class FormatPlug(Plug):
	node : SubdivToPoly = None
	pass
class InSubdCVIdLeftPlug(Plug):
	parent : InSubdCVIdPlug = PlugDescriptor("inSubdCVId")
	node : SubdivToPoly = None
	pass
class InSubdCVIdRightPlug(Plug):
	parent : InSubdCVIdPlug = PlugDescriptor("inSubdCVId")
	node : SubdivToPoly = None
	pass
class InSubdCVIdPlug(Plug):
	inSubdCVIdLeft_ : InSubdCVIdLeftPlug = PlugDescriptor("inSubdCVIdLeft")
	isl_ : InSubdCVIdLeftPlug = PlugDescriptor("inSubdCVIdLeft")
	inSubdCVIdRight_ : InSubdCVIdRightPlug = PlugDescriptor("inSubdCVIdRight")
	isr_ : InSubdCVIdRightPlug = PlugDescriptor("inSubdCVIdRight")
	node : SubdivToPoly = None
	pass
class InSubdivPlug(Plug):
	node : SubdivToPoly = None
	pass
class LevelPlug(Plug):
	node : SubdivToPoly = None
	pass
class MaxPolysPlug(Plug):
	node : SubdivToPoly = None
	pass
class OutMeshPlug(Plug):
	node : SubdivToPoly = None
	pass
class OutSubdCVIdLeftPlug(Plug):
	parent : OutSubdCVIdPlug = PlugDescriptor("outSubdCVId")
	node : SubdivToPoly = None
	pass
class OutSubdCVIdRightPlug(Plug):
	parent : OutSubdCVIdPlug = PlugDescriptor("outSubdCVId")
	node : SubdivToPoly = None
	pass
class OutSubdCVIdPlug(Plug):
	outSubdCVIdLeft_ : OutSubdCVIdLeftPlug = PlugDescriptor("outSubdCVIdLeft")
	osl_ : OutSubdCVIdLeftPlug = PlugDescriptor("outSubdCVIdLeft")
	outSubdCVIdRight_ : OutSubdCVIdRightPlug = PlugDescriptor("outSubdCVIdRight")
	osr_ : OutSubdCVIdRightPlug = PlugDescriptor("outSubdCVIdRight")
	node : SubdivToPoly = None
	pass
class OutvPlug(Plug):
	node : SubdivToPoly = None
	pass
class PolygonTypePlug(Plug):
	node : SubdivToPoly = None
	pass
class PreserveVertexOrderingPlug(Plug):
	node : SubdivToPoly = None
	pass
class SampleCountPlug(Plug):
	node : SubdivToPoly = None
	pass
class ShareUVsPlug(Plug):
	node : SubdivToPoly = None
	pass
class SubdNormalsPlug(Plug):
	node : SubdivToPoly = None
	pass
# endregion


# define node class
class SubdivToPoly(_BASE_):
	applyMatrixToResult_ : ApplyMatrixToResultPlug = PlugDescriptor("applyMatrixToResult")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	convertComp_ : ConvertCompPlug = PlugDescriptor("convertComp")
	copyUVTopology_ : CopyUVTopologyPlug = PlugDescriptor("copyUVTopology")
	depth_ : DepthPlug = PlugDescriptor("depth")
	extractPointPosition_ : ExtractPointPositionPlug = PlugDescriptor("extractPointPosition")
	format_ : FormatPlug = PlugDescriptor("format")
	inSubdCVIdLeft_ : InSubdCVIdLeftPlug = PlugDescriptor("inSubdCVIdLeft")
	inSubdCVIdRight_ : InSubdCVIdRightPlug = PlugDescriptor("inSubdCVIdRight")
	inSubdCVId_ : InSubdCVIdPlug = PlugDescriptor("inSubdCVId")
	inSubdiv_ : InSubdivPlug = PlugDescriptor("inSubdiv")
	level_ : LevelPlug = PlugDescriptor("level")
	maxPolys_ : MaxPolysPlug = PlugDescriptor("maxPolys")
	outMesh_ : OutMeshPlug = PlugDescriptor("outMesh")
	outSubdCVIdLeft_ : OutSubdCVIdLeftPlug = PlugDescriptor("outSubdCVIdLeft")
	outSubdCVIdRight_ : OutSubdCVIdRightPlug = PlugDescriptor("outSubdCVIdRight")
	outSubdCVId_ : OutSubdCVIdPlug = PlugDescriptor("outSubdCVId")
	outv_ : OutvPlug = PlugDescriptor("outv")
	polygonType_ : PolygonTypePlug = PlugDescriptor("polygonType")
	preserveVertexOrdering_ : PreserveVertexOrderingPlug = PlugDescriptor("preserveVertexOrdering")
	sampleCount_ : SampleCountPlug = PlugDescriptor("sampleCount")
	shareUVs_ : ShareUVsPlug = PlugDescriptor("shareUVs")
	subdNormals_ : SubdNormalsPlug = PlugDescriptor("subdNormals")

	# node attributes

	typeName = "subdivToPoly"
	apiTypeInt = 719
	apiTypeStr = "kSubdivToPoly"
	typeIdInt = 1396986704
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["applyMatrixToResult", "binMembership", "convertComp", "copyUVTopology", "depth", "extractPointPosition", "format", "inSubdCVIdLeft", "inSubdCVIdRight", "inSubdCVId", "inSubdiv", "level", "maxPolys", "outMesh", "outSubdCVIdLeft", "outSubdCVIdRight", "outSubdCVId", "outv", "polygonType", "preserveVertexOrdering", "sampleCount", "shareUVs", "subdNormals"]
	nodeLeafPlugs = ["applyMatrixToResult", "binMembership", "convertComp", "copyUVTopology", "depth", "extractPointPosition", "format", "inSubdCVId", "inSubdiv", "level", "maxPolys", "outMesh", "outSubdCVId", "outv", "polygonType", "preserveVertexOrdering", "sampleCount", "shareUVs", "subdNormals"]
	pass

