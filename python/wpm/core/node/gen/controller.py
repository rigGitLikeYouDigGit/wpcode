

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : Controller = None
	pass
class ChildrenPlug(Plug):
	node : Controller = None
	pass
class ControllerObjectPlug(Plug):
	node : Controller = None
	pass
class CycleWalkSiblingPlug(Plug):
	node : Controller = None
	pass
class ParentPlug(Plug):
	node : Controller = None
	pass
class ParentprepopulatePlug(Plug):
	node : Controller = None
	pass
class PrepopulatePlug(Plug):
	node : Controller = None
	pass
class VisibilityModePlug(Plug):
	node : Controller = None
	pass
# endregion


# define node class
class Controller(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	children_ : ChildrenPlug = PlugDescriptor("children")
	controllerObject_ : ControllerObjectPlug = PlugDescriptor("controllerObject")
	cycleWalkSibling_ : CycleWalkSiblingPlug = PlugDescriptor("cycleWalkSibling")
	parent_ : ParentPlug = PlugDescriptor("parent")
	parentprepopulate_ : ParentprepopulatePlug = PlugDescriptor("parentprepopulate")
	prepopulate_ : PrepopulatePlug = PlugDescriptor("prepopulate")
	visibilityMode_ : VisibilityModePlug = PlugDescriptor("visibilityMode")

	# node attributes

	typeName = "controller"
	typeIdInt = 1128747600
	nodeLeafClassAttrs = ["binMembership", "children", "controllerObject", "cycleWalkSibling", "parent", "parentprepopulate", "prepopulate", "visibilityMode"]
	nodeLeafPlugs = ["binMembership", "children", "controllerObject", "cycleWalkSibling", "parent", "parentprepopulate", "prepopulate", "visibilityMode"]
	pass

