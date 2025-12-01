

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ObjectFilter = retriever.getNodeCls("ObjectFilter")
assert ObjectFilter
if T.TYPE_CHECKING:
	from .. import ObjectFilter

# add node doc



# region plug type defs
class ArrayArgPlug(Plug):
	node : ObjectScriptFilter = None
	pass
class AttrNamePlug(Plug):
	node : ObjectScriptFilter = None
	pass
class ProcNamePlug(Plug):
	node : ObjectScriptFilter = None
	pass
class PythonModulePlug(Plug):
	node : ObjectScriptFilter = None
	pass
class UniqueNodeNamesPlug(Plug):
	node : ObjectScriptFilter = None
	pass
# endregion


# define node class
class ObjectScriptFilter(ObjectFilter):
	arrayArg_ : ArrayArgPlug = PlugDescriptor("arrayArg")
	attrName_ : AttrNamePlug = PlugDescriptor("attrName")
	procName_ : ProcNamePlug = PlugDescriptor("procName")
	pythonModule_ : PythonModulePlug = PlugDescriptor("pythonModule")
	uniqueNodeNames_ : UniqueNodeNamesPlug = PlugDescriptor("uniqueNodeNames")

	# node attributes

	typeName = "objectScriptFilter"
	apiTypeInt = 682
	apiTypeStr = "kObjectScriptFilter"
	typeIdInt = 1330857548
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["arrayArg", "attrName", "procName", "pythonModule", "uniqueNodeNames"]
	nodeLeafPlugs = ["arrayArg", "attrName", "procName", "pythonModule", "uniqueNodeNames"]
	pass

