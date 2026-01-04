

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
	node : DynGlobals = None
	pass
class CacheDirectoryPlug(Plug):
	node : DynGlobals = None
	pass
class ConfirmSceneNamePlug(Plug):
	node : DynGlobals = None
	pass
class ConfirmedPathPlug(Plug):
	node : DynGlobals = None
	pass
class InternalOverSamplesPlug(Plug):
	node : DynGlobals = None
	pass
class MaxFrameCachedPlug(Plug):
	node : DynGlobals = None
	pass
class MinFrameCachedPlug(Plug):
	node : DynGlobals = None
	pass
class OverSamplesPlug(Plug):
	node : DynGlobals = None
	pass
class UseParticleDiskCachePlug(Plug):
	node : DynGlobals = None
	pass
# endregion


# define node class
class DynGlobals(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	cacheDirectory_ : CacheDirectoryPlug = PlugDescriptor("cacheDirectory")
	confirmSceneName_ : ConfirmSceneNamePlug = PlugDescriptor("confirmSceneName")
	confirmedPath_ : ConfirmedPathPlug = PlugDescriptor("confirmedPath")
	internalOverSamples_ : InternalOverSamplesPlug = PlugDescriptor("internalOverSamples")
	maxFrameCached_ : MaxFrameCachedPlug = PlugDescriptor("maxFrameCached")
	minFrameCached_ : MinFrameCachedPlug = PlugDescriptor("minFrameCached")
	overSamples_ : OverSamplesPlug = PlugDescriptor("overSamples")
	useParticleDiskCache_ : UseParticleDiskCachePlug = PlugDescriptor("useParticleDiskCache")

	# node attributes

	typeName = "dynGlobals"
	apiTypeInt = 769
	apiTypeStr = "kDynGlobals"
	typeIdInt = 1497646924
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "cacheDirectory", "confirmSceneName", "confirmedPath", "internalOverSamples", "maxFrameCached", "minFrameCached", "overSamples", "useParticleDiskCache"]
	nodeLeafPlugs = ["binMembership", "cacheDirectory", "confirmSceneName", "confirmedPath", "internalOverSamples", "maxFrameCached", "minFrameCached", "overSamples", "useParticleDiskCache"]
	pass

