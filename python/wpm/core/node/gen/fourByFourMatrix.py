

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
	node : FourByFourMatrix = None
	pass
class In00Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In01Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In02Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In03Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In10Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In11Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In12Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In13Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In20Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In21Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In22Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In23Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In30Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In31Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In32Plug(Plug):
	node : FourByFourMatrix = None
	pass
class In33Plug(Plug):
	node : FourByFourMatrix = None
	pass
class OutputPlug(Plug):
	node : FourByFourMatrix = None
	pass
# endregion


# define node class
class FourByFourMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	in00_ : In00Plug = PlugDescriptor("in00")
	in01_ : In01Plug = PlugDescriptor("in01")
	in02_ : In02Plug = PlugDescriptor("in02")
	in03_ : In03Plug = PlugDescriptor("in03")
	in10_ : In10Plug = PlugDescriptor("in10")
	in11_ : In11Plug = PlugDescriptor("in11")
	in12_ : In12Plug = PlugDescriptor("in12")
	in13_ : In13Plug = PlugDescriptor("in13")
	in20_ : In20Plug = PlugDescriptor("in20")
	in21_ : In21Plug = PlugDescriptor("in21")
	in22_ : In22Plug = PlugDescriptor("in22")
	in23_ : In23Plug = PlugDescriptor("in23")
	in30_ : In30Plug = PlugDescriptor("in30")
	in31_ : In31Plug = PlugDescriptor("in31")
	in32_ : In32Plug = PlugDescriptor("in32")
	in33_ : In33Plug = PlugDescriptor("in33")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "fourByFourMatrix"
	apiTypeInt = 775
	apiTypeStr = "kFourByFourMatrix"
	typeIdInt = 1178748493
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "in00", "in01", "in02", "in03", "in10", "in11", "in12", "in13", "in20", "in21", "in22", "in23", "in30", "in31", "in32", "in33", "output"]
	nodeLeafPlugs = ["binMembership", "in00", "in01", "in02", "in03", "in10", "in11", "in12", "in13", "in20", "in21", "in22", "in23", "in30", "in31", "in32", "in33", "output"]
	pass

