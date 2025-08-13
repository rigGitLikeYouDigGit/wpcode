

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ManipContainer = retriever.getNodeCls("ManipContainer")
assert ManipContainer
if T.TYPE_CHECKING:
	from .. import ManipContainer

# add node doc



# region plug type defs
class AbovePlug(Plug):
	node : JointClusterManip = None
	pass
class AboveJointInstancePlug(Plug):
	node : JointClusterManip = None
	pass
class BelowJointInstancePlug(Plug):
	node : JointClusterManip = None
	pass
# endregion


# define node class
class JointClusterManip(ManipContainer):
	above_ : AbovePlug = PlugDescriptor("above")
	aboveJointInstance_ : AboveJointInstancePlug = PlugDescriptor("aboveJointInstance")
	belowJointInstance_ : BelowJointInstancePlug = PlugDescriptor("belowJointInstance")

	# node attributes

	typeName = "jointClusterManip"
	apiTypeInt = 168
	apiTypeStr = "kJointClusterManip"
	typeIdInt = 1431128643
	MFnCls = om.MFnManip3D
	pass

