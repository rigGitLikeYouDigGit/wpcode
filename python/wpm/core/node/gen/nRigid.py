

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	NBase = Catalogue.NBase
else:
	from .. import retriever
	NBase = retriever.getNodeCls("NBase")
	assert NBase

# add node doc



# region plug type defs
class SolverDisplayPlug(Plug):
	node : NRigid = None
	pass
# endregion


# define node class
class NRigid(NBase):
	solverDisplay_ : SolverDisplayPlug = PlugDescriptor("solverDisplay")

	# node attributes

	typeName = "nRigid"
	apiTypeInt = 1009
	apiTypeStr = "kNRigid"
	typeIdInt = 1314015044
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["solverDisplay"]
	nodeLeafPlugs = ["solverDisplay"]
	pass

