

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class DeformerDataPlug(Plug):
	node : DeformFunc = None
	pass
class HandleWidthPlug(Plug):
	node : DeformFunc = None
	pass
# endregion


# define node class
class DeformFunc(Shape):
	deformerData_ : DeformerDataPlug = PlugDescriptor("deformerData")
	handleWidth_ : HandleWidthPlug = PlugDescriptor("handleWidth")

	# node attributes

	typeName = "deformFunc"
	apiTypeInt = 624
	apiTypeStr = "kDeformFunc"
	typeIdInt = 1178879558
	MFnCls = om.MFnDagNode
	pass

