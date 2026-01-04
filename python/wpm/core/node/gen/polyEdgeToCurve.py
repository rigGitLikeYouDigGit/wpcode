

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs
class ConformToSmoothMeshPreviewPlug(Plug):
	node : PolyEdgeToCurve = None
	pass
class DegreePlug(Plug):
	node : PolyEdgeToCurve = None
	pass
class DisplaySmoothMeshPlug(Plug):
	node : PolyEdgeToCurve = None
	pass
class FormPlug(Plug):
	node : PolyEdgeToCurve = None
	pass
class InputComponentsPlug(Plug):
	node : PolyEdgeToCurve = None
	pass
class InputMatPlug(Plug):
	node : PolyEdgeToCurve = None
	pass
class InputPolymeshPlug(Plug):
	node : PolyEdgeToCurve = None
	pass
class InputSmoothPolymeshPlug(Plug):
	node : PolyEdgeToCurve = None
	pass
class OutputcurvePlug(Plug):
	node : PolyEdgeToCurve = None
	pass
class SmoothLevelPlug(Plug):
	node : PolyEdgeToCurve = None
	pass
# endregion


# define node class
class PolyEdgeToCurve(AbstractBaseCreate):
	conformToSmoothMeshPreview_ : ConformToSmoothMeshPreviewPlug = PlugDescriptor("conformToSmoothMeshPreview")
	degree_ : DegreePlug = PlugDescriptor("degree")
	displaySmoothMesh_ : DisplaySmoothMeshPlug = PlugDescriptor("displaySmoothMesh")
	form_ : FormPlug = PlugDescriptor("form")
	inputComponents_ : InputComponentsPlug = PlugDescriptor("inputComponents")
	inputMat_ : InputMatPlug = PlugDescriptor("inputMat")
	inputPolymesh_ : InputPolymeshPlug = PlugDescriptor("inputPolymesh")
	inputSmoothPolymesh_ : InputSmoothPolymeshPlug = PlugDescriptor("inputSmoothPolymesh")
	outputcurve_ : OutputcurvePlug = PlugDescriptor("outputcurve")
	smoothLevel_ : SmoothLevelPlug = PlugDescriptor("smoothLevel")

	# node attributes

	typeName = "polyEdgeToCurve"
	apiTypeInt = 1019
	apiTypeStr = "kPolyEdgeToCurve"
	typeIdInt = 1347699542
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["conformToSmoothMeshPreview", "degree", "displaySmoothMesh", "form", "inputComponents", "inputMat", "inputPolymesh", "inputSmoothPolymesh", "outputcurve", "smoothLevel"]
	nodeLeafPlugs = ["conformToSmoothMeshPreview", "degree", "displaySmoothMesh", "form", "inputComponents", "inputMat", "inputPolymesh", "inputSmoothPolymesh", "outputcurve", "smoothLevel"]
	pass

