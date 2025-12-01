

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyCreator = retriever.getNodeCls("PolyCreator")
assert PolyCreator
if T.TYPE_CHECKING:
	from .. import PolyCreator

# add node doc



# region plug type defs
class ComponentTagNamePlug(Plug):
	node : PolyUnite = None
	pass
class InputMatPlug(Plug):
	node : PolyUnite = None
	pass
class InputPolyPlug(Plug):
	node : PolyUnite = None
	pass
class MergeUVSetsPlug(Plug):
	node : PolyUnite = None
	pass
class OutputUVSetNamePlug(Plug):
	node : PolyUnite = None
	pass
class UseOldPolyArchitecturePlug(Plug):
	node : PolyUnite = None
	pass
# endregion


# define node class
class PolyUnite(PolyCreator):
	componentTagName_ : ComponentTagNamePlug = PlugDescriptor("componentTagName")
	inputMat_ : InputMatPlug = PlugDescriptor("inputMat")
	inputPoly_ : InputPolyPlug = PlugDescriptor("inputPoly")
	mergeUVSets_ : MergeUVSetsPlug = PlugDescriptor("mergeUVSets")
	outputUVSetName_ : OutputUVSetNamePlug = PlugDescriptor("outputUVSetName")
	useOldPolyArchitecture_ : UseOldPolyArchitecturePlug = PlugDescriptor("useOldPolyArchitecture")

	# node attributes

	typeName = "polyUnite"
	apiTypeInt = 444
	apiTypeStr = "kPolyUnite"
	typeIdInt = 1347767881
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["componentTagName", "inputMat", "inputPoly", "mergeUVSets", "outputUVSetName", "useOldPolyArchitecture"]
	nodeLeafPlugs = ["componentTagName", "inputMat", "inputPoly", "mergeUVSets", "outputUVSetName", "useOldPolyArchitecture"]
	pass

