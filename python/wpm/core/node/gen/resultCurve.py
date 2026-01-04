

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AnimCurve = Catalogue.AnimCurve
else:
	from .. import retriever
	AnimCurve = retriever.getNodeCls("AnimCurve")
	assert AnimCurve

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
	nodeLeafClassAttrs = ["end", "sampleBy", "start"]
	nodeLeafPlugs = ["end", "sampleBy", "start"]
	pass

