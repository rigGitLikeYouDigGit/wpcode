

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Selector = retriever.getNodeCls("Selector")
assert Selector
if T.TYPE_CHECKING:
	from .. import Selector

# add node doc



# region plug type defs
class CustomFilterValuePlug(Plug):
	node : SimpleSelector = None
	pass
class PatternPlug(Plug):
	node : SimpleSelector = None
	pass
class PreviousPatternPlug(Plug):
	node : SimpleSelector = None
	pass
class StaticSelectionPlug(Plug):
	node : SimpleSelector = None
	pass
class TypeFilterPlug(Plug):
	node : SimpleSelector = None
	pass
# endregion


# define node class
class SimpleSelector(Selector):
	customFilterValue_ : CustomFilterValuePlug = PlugDescriptor("customFilterValue")
	pattern_ : PatternPlug = PlugDescriptor("pattern")
	previousPattern_ : PreviousPatternPlug = PlugDescriptor("previousPattern")
	staticSelection_ : StaticSelectionPlug = PlugDescriptor("staticSelection")
	typeFilter_ : TypeFilterPlug = PlugDescriptor("typeFilter")

	# node attributes

	typeName = "simpleSelector"
	typeIdInt = 1476395934
	pass

