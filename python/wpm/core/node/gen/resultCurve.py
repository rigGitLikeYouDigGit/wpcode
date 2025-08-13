

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AnimCurve = retriever.getNodeCls("AnimCurve")
assert AnimCurve
if T.TYPE_CHECKING:
	from .. import AnimCurve

# add node doc



# region plug type defs
class EndPlug(Plug):
	node : ResultCurve = None
	pass
class SampleByPlug(Plug):
	node : ResultCurve = None
	pass
class StartPlug(Plug):
	node : ResultCurve = None
	pass
# endregion


# define node class
class ResultCurve(AnimCurve):
	end_ : EndPlug = PlugDescriptor("end")
	sampleBy_ : SampleByPlug = PlugDescriptor("sampleBy")
	start_ : StartPlug = PlugDescriptor("start")

	# node attributes

	typeName = "resultCurve"
	apiTypeInt = 16
	apiTypeStr = "kResultCurve"
	typeIdInt = 1381188438
	MFnCls = om.MFnAnimCurve
	pass

