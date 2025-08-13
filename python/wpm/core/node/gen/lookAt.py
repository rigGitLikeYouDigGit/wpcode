

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AimConstraint = retriever.getNodeCls("AimConstraint")
assert AimConstraint
if T.TYPE_CHECKING:
	from .. import AimConstraint

# add node doc



# region plug type defs
class DisplayConnectorPlug(Plug):
	node : LookAt = None
	pass
class DistanceBetweenPlug(Plug):
	node : LookAt = None
	pass
class TwistPlug(Plug):
	node : LookAt = None
	pass
# endregion


# define node class
class LookAt(AimConstraint):
	displayConnector_ : DisplayConnectorPlug = PlugDescriptor("displayConnector")
	distanceBetween_ : DistanceBetweenPlug = PlugDescriptor("distanceBetween")
	twist_ : TwistPlug = PlugDescriptor("twist")

	# node attributes

	typeName = "lookAt"
	apiTypeInt = 112
	apiTypeStr = "kLookAt"
	typeIdInt = 1145848148
	MFnCls = om.MFnTransform
	pass

