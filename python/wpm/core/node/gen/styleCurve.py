

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
class OutCurvePlug(Plug):
	node : StyleCurve = None
	pass
class StylePlug(Plug):
	node : StyleCurve = None
	pass
# endregion


# define node class
class StyleCurve(AbstractBaseCreate):
	outCurve_ : OutCurvePlug = PlugDescriptor("outCurve")
	style_ : StylePlug = PlugDescriptor("style")

	# node attributes

	typeName = "styleCurve"
	apiTypeInt = 900
	apiTypeStr = "kStyleCurve"
	typeIdInt = 1314083907
	MFnCls = om.MFnDependencyNode
	pass

