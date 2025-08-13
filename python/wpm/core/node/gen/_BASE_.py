

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports


# add node doc



# region plug type defs
class CachingPlug(Plug):
	node : _BASE_ = None
	pass
class FrozenPlug(Plug):
	node : _BASE_ = None
	pass
class IsHistoricallyInterestingPlug(Plug):
	node : _BASE_ = None
	pass
class MessagePlug(Plug):
	node : _BASE_ = None
	pass
class NodeStatePlug(Plug):
	node : _BASE_ = None
	pass
# endregion


# define node class
class _BASE_(WN):
	caching_ : CachingPlug = PlugDescriptor("caching")
	frozen_ : FrozenPlug = PlugDescriptor("frozen")
	isHistoricallyInteresting_ : IsHistoricallyInterestingPlug = PlugDescriptor("isHistoricallyInteresting")
	message_ : MessagePlug = PlugDescriptor("message")
	nodeState_ : NodeStatePlug = PlugDescriptor("nodeState")

	# node attributes

	typeName = "_BASE_"
	typeIdInt = -1
	MFnCls = om.MFnDependencyNode
	pass

