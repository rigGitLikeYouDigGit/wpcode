

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : LightLinker = None
	pass
class LightIgnoredPlug(Plug):
	parent : IgnorePlug = PlugDescriptor("ignore")
	node : LightLinker = None
	pass
class ObjectIgnoredPlug(Plug):
	parent : IgnorePlug = PlugDescriptor("ignore")
	node : LightLinker = None
	pass
class IgnorePlug(Plug):
	lightIgnored_ : LightIgnoredPlug = PlugDescriptor("lightIgnored")
	lign_ : LightIgnoredPlug = PlugDescriptor("lightIgnored")
	objectIgnored_ : ObjectIgnoredPlug = PlugDescriptor("objectIgnored")
	oign_ : ObjectIgnoredPlug = PlugDescriptor("objectIgnored")
	node : LightLinker = None
	pass
class LightPlug(Plug):
	parent : LinkPlug = PlugDescriptor("link")
	node : LightLinker = None
	pass
class ObjectPlug(Plug):
	parent : LinkPlug = PlugDescriptor("link")
	node : LightLinker = None
	pass
class LinkPlug(Plug):
	light_ : LightPlug = PlugDescriptor("light")
	llnk_ : LightPlug = PlugDescriptor("light")
	object_ : ObjectPlug = PlugDescriptor("object")
	olnk_ : ObjectPlug = PlugDescriptor("object")
	node : LightLinker = None
	pass
class ShadowLightIgnoredPlug(Plug):
	parent : ShadowIgnorePlug = PlugDescriptor("shadowIgnore")
	node : LightLinker = None
	pass
class ShadowObjectIgnoredPlug(Plug):
	parent : ShadowIgnorePlug = PlugDescriptor("shadowIgnore")
	node : LightLinker = None
	pass
class ShadowIgnorePlug(Plug):
	shadowLightIgnored_ : ShadowLightIgnoredPlug = PlugDescriptor("shadowLightIgnored")
	slig_ : ShadowLightIgnoredPlug = PlugDescriptor("shadowLightIgnored")
	shadowObjectIgnored_ : ShadowObjectIgnoredPlug = PlugDescriptor("shadowObjectIgnored")
	soig_ : ShadowObjectIgnoredPlug = PlugDescriptor("shadowObjectIgnored")
	node : LightLinker = None
	pass
class ShadowLightPlug(Plug):
	parent : ShadowLinkPlug = PlugDescriptor("shadowLink")
	node : LightLinker = None
	pass
class ShadowObjectPlug(Plug):
	parent : ShadowLinkPlug = PlugDescriptor("shadowLink")
	node : LightLinker = None
	pass
class ShadowLinkPlug(Plug):
	shadowLight_ : ShadowLightPlug = PlugDescriptor("shadowLight")
	sllk_ : ShadowLightPlug = PlugDescriptor("shadowLight")
	shadowObject_ : ShadowObjectPlug = PlugDescriptor("shadowObject")
	solk_ : ShadowObjectPlug = PlugDescriptor("shadowObject")
	node : LightLinker = None
	pass
# endregion


# define node class
class LightLinker(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	lightIgnored_ : LightIgnoredPlug = PlugDescriptor("lightIgnored")
	objectIgnored_ : ObjectIgnoredPlug = PlugDescriptor("objectIgnored")
	ignore_ : IgnorePlug = PlugDescriptor("ignore")
	light_ : LightPlug = PlugDescriptor("light")
	object_ : ObjectPlug = PlugDescriptor("object")
	link_ : LinkPlug = PlugDescriptor("link")
	shadowLightIgnored_ : ShadowLightIgnoredPlug = PlugDescriptor("shadowLightIgnored")
	shadowObjectIgnored_ : ShadowObjectIgnoredPlug = PlugDescriptor("shadowObjectIgnored")
	shadowIgnore_ : ShadowIgnorePlug = PlugDescriptor("shadowIgnore")
	shadowLight_ : ShadowLightPlug = PlugDescriptor("shadowLight")
	shadowObject_ : ShadowObjectPlug = PlugDescriptor("shadowObject")
	shadowLink_ : ShadowLinkPlug = PlugDescriptor("shadowLink")

	# node attributes

	typeName = "lightLinker"
	typeIdInt = 1380731979
	pass

