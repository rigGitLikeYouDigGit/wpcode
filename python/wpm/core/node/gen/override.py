

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ChildNode = retriever.getNodeCls("ChildNode")
assert ChildNode
if T.TYPE_CHECKING:
	from .. import ChildNode

# add node doc



# region plug type defs
class AttributePlug(Plug):
	node : Override = None
	pass
class EnabledPlug(Plug):
	node : Override = None
	pass
class LocalRenderPlug(Plug):
	node : Override = None
	pass
class ParentEnabledPlug(Plug):
	node : Override = None
	pass
class ParentNumIsolatedChildrenPlug(Plug):
	node : Override = None
	pass
class SelfEnabledPlug(Plug):
	node : Override = None
	pass
# endregion


# define node class
class Override(ChildNode):
	attribute_ : AttributePlug = PlugDescriptor("attribute")
	enabled_ : EnabledPlug = PlugDescriptor("enabled")
	localRender_ : LocalRenderPlug = PlugDescriptor("localRender")
	parentEnabled_ : ParentEnabledPlug = PlugDescriptor("parentEnabled")
	parentNumIsolatedChildren_ : ParentNumIsolatedChildrenPlug = PlugDescriptor("parentNumIsolatedChildren")
	selfEnabled_ : SelfEnabledPlug = PlugDescriptor("selfEnabled")

	# node attributes

	typeName = "override"
	typeIdInt = 1476395888
	pass

