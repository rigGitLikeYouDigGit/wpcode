

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Reflect = retriever.getNodeCls("Reflect")
assert Reflect
if T.TYPE_CHECKING:
	from .. import Reflect

# add node doc



# region plug type defs
class HighlightSizePlug(Plug):
	node : PhongE = None
	pass
class RoughnessPlug(Plug):
	node : PhongE = None
	pass
class WhitenessBPlug(Plug):
	parent : WhitenessPlug = PlugDescriptor("whiteness")
	node : PhongE = None
	pass
class WhitenessGPlug(Plug):
	parent : WhitenessPlug = PlugDescriptor("whiteness")
	node : PhongE = None
	pass
class WhitenessRPlug(Plug):
	parent : WhitenessPlug = PlugDescriptor("whiteness")
	node : PhongE = None
	pass
class WhitenessPlug(Plug):
	whitenessB_ : WhitenessBPlug = PlugDescriptor("whitenessB")
	wnb_ : WhitenessBPlug = PlugDescriptor("whitenessB")
	whitenessG_ : WhitenessGPlug = PlugDescriptor("whitenessG")
	wng_ : WhitenessGPlug = PlugDescriptor("whitenessG")
	whitenessR_ : WhitenessRPlug = PlugDescriptor("whitenessR")
	wnr_ : WhitenessRPlug = PlugDescriptor("whitenessR")
	node : PhongE = None
	pass
# endregion


# define node class
class PhongE(Reflect):
	highlightSize_ : HighlightSizePlug = PlugDescriptor("highlightSize")
	roughness_ : RoughnessPlug = PlugDescriptor("roughness")
	whitenessB_ : WhitenessBPlug = PlugDescriptor("whitenessB")
	whitenessG_ : WhitenessGPlug = PlugDescriptor("whitenessG")
	whitenessR_ : WhitenessRPlug = PlugDescriptor("whitenessR")
	whiteness_ : WhitenessPlug = PlugDescriptor("whiteness")

	# node attributes

	typeName = "phongE"
	typeIdInt = 1380993093
	pass

