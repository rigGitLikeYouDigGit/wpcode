

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ParentTessellate = retriever.getNodeCls("ParentTessellate")
assert ParentTessellate
if T.TYPE_CHECKING:
	from .. import ParentTessellate

# add node doc



# region plug type defs
class InputShellPlug(Plug):
	node : ShellTessellate = None
	pass
# endregion


# define node class
class ShellTessellate(ParentTessellate):
	inputShell_ : InputShellPlug = PlugDescriptor("inputShell")

	# node attributes

	typeName = "shellTessellate"
	typeIdInt = 1398031699
	nodeLeafClassAttrs = ["inputShell"]
	nodeLeafPlugs = ["inputShell"]
	pass

