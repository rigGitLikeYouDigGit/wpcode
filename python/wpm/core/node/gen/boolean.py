

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
class InputShellAPlug(Plug):
	node : Boolean = None
	pass
class InputShellBPlug(Plug):
	node : Boolean = None
	pass
class OperationPlug(Plug):
	node : Boolean = None
	pass
class OutputShellPlug(Plug):
	node : Boolean = None
	pass
class TolerancePlug(Plug):
	node : Boolean = None
	pass
# endregion


# define node class
class Boolean(AbstractBaseCreate):
	inputShellA_ : InputShellAPlug = PlugDescriptor("inputShellA")
	inputShellB_ : InputShellBPlug = PlugDescriptor("inputShellB")
	operation_ : OperationPlug = PlugDescriptor("operation")
	outputShell_ : OutputShellPlug = PlugDescriptor("outputShell")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "boolean"
	typeIdInt = 1312968524
	pass

