

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class InputShellPlug(Plug):
	node : ExplodeNurbsShell = None
	pass
class OutputSurfacePlug(Plug):
	node : ExplodeNurbsShell = None
	pass
# endregion


# define node class
class ExplodeNurbsShell(AbstractBaseCreate):
	inputShell_ : InputShellPlug = PlugDescriptor("inputShell")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")

	# node attributes

	typeName = "explodeNurbsShell"
	apiTypeInt = 692
	apiTypeStr = "kExplodeNurbsShell"
	typeIdInt = 1313166152
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["inputShell", "outputSurface"]
	nodeLeafPlugs = ["inputShell", "outputSurface"]
	pass

