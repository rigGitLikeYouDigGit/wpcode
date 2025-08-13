

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
class IndexPlug(Plug):
	node : HardenPoint = None
	pass
class InputCurvePlug(Plug):
	node : HardenPoint = None
	pass
class MultiplicityPlug(Plug):
	node : HardenPoint = None
	pass
class OutputCurvePlug(Plug):
	node : HardenPoint = None
	pass
# endregion


# define node class
class HardenPoint(AbstractBaseCreate):
	index_ : IndexPlug = PlugDescriptor("index")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	multiplicity_ : MultiplicityPlug = PlugDescriptor("multiplicity")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")

	# node attributes

	typeName = "hardenPoint"
	typeIdInt = 1313358928
	pass

