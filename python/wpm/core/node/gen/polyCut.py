

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs
class CompIdPlug(Plug):
	node : PolyCut = None
	pass
class CutPlaneCenterXPlug(Plug):
	parent : CutPlaneCenterPlug = PlugDescriptor("cutPlaneCenter")
	node : PolyCut = None
	pass
class CutPlaneCenterYPlug(Plug):
	parent : CutPlaneCenterPlug = PlugDescriptor("cutPlaneCenter")
	node : PolyCut = None
	pass
class CutPlaneCenterZPlug(Plug):
	parent : CutPlaneCenterPlug = PlugDescriptor("cutPlaneCenter")
	node : PolyCut = None
	pass
class CutPlaneCenterPlug(Plug):
	cutPlaneCenterX_ : CutPlaneCenterXPlug = PlugDescriptor("cutPlaneCenterX")
	pcx_ : CutPlaneCenterXPlug = PlugDescriptor("cutPlaneCenterX")
	cutPlaneCenterY_ : CutPlaneCenterYPlug = PlugDescriptor("cutPlaneCenterY")
	pcy_ : CutPlaneCenterYPlug = PlugDescriptor("cutPlaneCenterY")
	cutPlaneCenterZ_ : CutPlaneCenterZPlug = PlugDescriptor("cutPlaneCenterZ")
	pcz_ : CutPlaneCenterZPlug = PlugDescriptor("cutPlaneCenterZ")
	node : PolyCut = None
	pass
class CutPlaneRotateXPlug(Plug):
	parent : CutPlaneRotatePlug = PlugDescriptor("cutPlaneRotate")
	node : PolyCut = None
	pass
class CutPlaneRotateYPlug(Plug):
	parent : CutPlaneRotatePlug = PlugDescriptor("cutPlaneRotate")
	node : PolyCut = None
	pass
class CutPlaneRotateZPlug(Plug):
	parent : CutPlaneRotatePlug = PlugDescriptor("cutPlaneRotate")
	node : PolyCut = None
	pass
class CutPlaneRotatePlug(Plug):
	cutPlaneRotateX_ : CutPlaneRotateXPlug = PlugDescriptor("cutPlaneRotateX")
	rx_ : CutPlaneRotateXPlug = PlugDescriptor("cutPlaneRotateX")
	cutPlaneRotateY_ : CutPlaneRotateYPlug = PlugDescriptor("cutPlaneRotateY")
	ry_ : CutPlaneRotateYPlug = PlugDescriptor("cutPlaneRotateY")
	cutPlaneRotateZ_ : CutPlaneRotateZPlug = PlugDescriptor("cutPlaneRotateZ")
	rz_ : CutPlaneRotateZPlug = PlugDescriptor("cutPlaneRotateZ")
	node : PolyCut = None
	pass
class CutPlaneHeightPlug(Plug):
	parent : CutPlaneSizePlug = PlugDescriptor("cutPlaneSize")
	node : PolyCut = None
	pass
class CutPlaneWidthPlug(Plug):
	parent : CutPlaneSizePlug = PlugDescriptor("cutPlaneSize")
	node : PolyCut = None
	pass
class CutPlaneSizePlug(Plug):
	cutPlaneHeight_ : CutPlaneHeightPlug = PlugDescriptor("cutPlaneHeight")
	ph_ : CutPlaneHeightPlug = PlugDescriptor("cutPlaneHeight")
	cutPlaneWidth_ : CutPlaneWidthPlug = PlugDescriptor("cutPlaneWidth")
	pw_ : CutPlaneWidthPlug = PlugDescriptor("cutPlaneWidth")
	node : PolyCut = None
	pass
class DeleteFacesPlug(Plug):
	node : PolyCut = None
	pass
class ExtractFacesPlug(Plug):
	node : PolyCut = None
	pass
class ExtractOffsetXPlug(Plug):
	parent : ExtractOffsetPlug = PlugDescriptor("extractOffset")
	node : PolyCut = None
	pass
class ExtractOffsetYPlug(Plug):
	parent : ExtractOffsetPlug = PlugDescriptor("extractOffset")
	node : PolyCut = None
	pass
class ExtractOffsetZPlug(Plug):
	parent : ExtractOffsetPlug = PlugDescriptor("extractOffset")
	node : PolyCut = None
	pass
class ExtractOffsetPlug(Plug):
	extractOffsetX_ : ExtractOffsetXPlug = PlugDescriptor("extractOffsetX")
	eox_ : ExtractOffsetXPlug = PlugDescriptor("extractOffsetX")
	extractOffsetY_ : ExtractOffsetYPlug = PlugDescriptor("extractOffsetY")
	eoy_ : ExtractOffsetYPlug = PlugDescriptor("extractOffsetY")
	extractOffsetZ_ : ExtractOffsetZPlug = PlugDescriptor("extractOffsetZ")
	eoz_ : ExtractOffsetZPlug = PlugDescriptor("extractOffsetZ")
	node : PolyCut = None
	pass
class OnObjectPlug(Plug):
	node : PolyCut = None
	pass
# endregion


# define node class
class PolyCut(PolyModifierWorld):
	compId_ : CompIdPlug = PlugDescriptor("compId")
	cutPlaneCenterX_ : CutPlaneCenterXPlug = PlugDescriptor("cutPlaneCenterX")
	cutPlaneCenterY_ : CutPlaneCenterYPlug = PlugDescriptor("cutPlaneCenterY")
	cutPlaneCenterZ_ : CutPlaneCenterZPlug = PlugDescriptor("cutPlaneCenterZ")
	cutPlaneCenter_ : CutPlaneCenterPlug = PlugDescriptor("cutPlaneCenter")
	cutPlaneRotateX_ : CutPlaneRotateXPlug = PlugDescriptor("cutPlaneRotateX")
	cutPlaneRotateY_ : CutPlaneRotateYPlug = PlugDescriptor("cutPlaneRotateY")
	cutPlaneRotateZ_ : CutPlaneRotateZPlug = PlugDescriptor("cutPlaneRotateZ")
	cutPlaneRotate_ : CutPlaneRotatePlug = PlugDescriptor("cutPlaneRotate")
	cutPlaneHeight_ : CutPlaneHeightPlug = PlugDescriptor("cutPlaneHeight")
	cutPlaneWidth_ : CutPlaneWidthPlug = PlugDescriptor("cutPlaneWidth")
	cutPlaneSize_ : CutPlaneSizePlug = PlugDescriptor("cutPlaneSize")
	deleteFaces_ : DeleteFacesPlug = PlugDescriptor("deleteFaces")
	extractFaces_ : ExtractFacesPlug = PlugDescriptor("extractFaces")
	extractOffsetX_ : ExtractOffsetXPlug = PlugDescriptor("extractOffsetX")
	extractOffsetY_ : ExtractOffsetYPlug = PlugDescriptor("extractOffsetY")
	extractOffsetZ_ : ExtractOffsetZPlug = PlugDescriptor("extractOffsetZ")
	extractOffset_ : ExtractOffsetPlug = PlugDescriptor("extractOffset")
	onObject_ : OnObjectPlug = PlugDescriptor("onObject")

	# node attributes

	typeName = "polyCut"
	apiTypeInt = 901
	apiTypeStr = "kPolyCut"
	typeIdInt = 1347437396
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["compId", "cutPlaneCenterX", "cutPlaneCenterY", "cutPlaneCenterZ", "cutPlaneCenter", "cutPlaneRotateX", "cutPlaneRotateY", "cutPlaneRotateZ", "cutPlaneRotate", "cutPlaneHeight", "cutPlaneWidth", "cutPlaneSize", "deleteFaces", "extractFaces", "extractOffsetX", "extractOffsetY", "extractOffsetZ", "extractOffset", "onObject"]
	nodeLeafPlugs = ["compId", "cutPlaneCenter", "cutPlaneRotate", "cutPlaneSize", "deleteFaces", "extractFaces", "extractOffset", "onObject"]
	pass

