

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
class InputSurfacePlug(Plug):
	node : StitchAsNurbsShell = None
	pass
class OutputShellPlug(Plug):
	node : StitchAsNurbsShell = None
	pass
class TolerancePlug(Plug):
	node : StitchAsNurbsShell = None
	pass
# endregion


# define node class
class StitchAsNurbsShell(AbstractBaseCreate):
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	outputShell_ : OutputShellPlug = PlugDescriptor("outputShell")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "stitchAsNurbsShell"
	apiTypeInt = 691
	apiTypeStr = "kStitchAsNurbsShell"
	typeIdInt = 1314083656
	MFnCls = om.MFnDependencyNode
	pass

