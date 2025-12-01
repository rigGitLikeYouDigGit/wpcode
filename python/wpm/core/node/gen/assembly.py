

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
DagContainer = retriever.getNodeCls("DagContainer")
assert DagContainer
if T.TYPE_CHECKING:
	from .. import DagContainer

# add node doc



# region plug type defs
class AssemblyEditsPlug(Plug):
	node : Assembly = None
	pass
# endregion


# define node class
class Assembly(DagContainer):
	assemblyEdits_ : AssemblyEditsPlug = PlugDescriptor("assemblyEdits")

	# node attributes

	typeName = "assembly"
	apiTypeInt = 1081
	apiTypeStr = "kAssembly"
	typeIdInt = 1095975513
	MFnCls = om.MFnAssembly
	nodeLeafClassAttrs = ["assemblyEdits"]
	nodeLeafPlugs = ["assemblyEdits"]
	pass

