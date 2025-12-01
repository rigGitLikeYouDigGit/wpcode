

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
CacheBase = retriever.getNodeCls("CacheBase")
assert CacheBase
if T.TYPE_CHECKING:
	from .. import CacheBase

# add node doc



# region plug type defs
class EndPlug(Plug):
	parent : CacheDataPlug = PlugDescriptor("cacheData")
	node : CacheBlend = None
	pass
class RangePlug(Plug):
	parent : CacheDataPlug = PlugDescriptor("cacheData")
	node : CacheBlend = None
	pass
class StartPlug(Plug):
	parent : CacheDataPlug = PlugDescriptor("cacheData")
	node : CacheBlend = None
	pass
class WeightPlug(Plug):
	parent : CacheDataPlug = PlugDescriptor("cacheData")
	node : CacheBlend = None
	pass
class CacheDataPlug(Plug):
	end_ : EndPlug = PlugDescriptor("end")
	e_ : EndPlug = PlugDescriptor("end")
	range_ : RangePlug = PlugDescriptor("range")
	ra_ : RangePlug = PlugDescriptor("range")
	start_ : StartPlug = PlugDescriptor("start")
	st_ : StartPlug = PlugDescriptor("start")
	weight_ : WeightPlug = PlugDescriptor("weight")
	w_ : WeightPlug = PlugDescriptor("weight")
	node : CacheBlend = None
	pass
class DisableAllPlug(Plug):
	node : CacheBlend = None
	pass
class PerPtWeightsPlug(Plug):
	parent : InCachePlug = PlugDescriptor("inCache")
	node : CacheBlend = None
	pass
class VectorArrayPlug(Plug):
	parent : InCachePlug = PlugDescriptor("inCache")
	node : CacheBlend = None
	pass
class InCachePlug(Plug):
	perPtWeights_ : PerPtWeightsPlug = PlugDescriptor("perPtWeights")
	ppw_ : PerPtWeightsPlug = PlugDescriptor("perPtWeights")
	vectorArray_ : VectorArrayPlug = PlugDescriptor("vectorArray")
	va_ : VectorArrayPlug = PlugDescriptor("vectorArray")
	node : CacheBlend = None
	pass
# endregion


# define node class
class CacheBlend(CacheBase):
	end_ : EndPlug = PlugDescriptor("end")
	range_ : RangePlug = PlugDescriptor("range")
	start_ : StartPlug = PlugDescriptor("start")
	weight_ : WeightPlug = PlugDescriptor("weight")
	cacheData_ : CacheDataPlug = PlugDescriptor("cacheData")
	disableAll_ : DisableAllPlug = PlugDescriptor("disableAll")
	perPtWeights_ : PerPtWeightsPlug = PlugDescriptor("perPtWeights")
	vectorArray_ : VectorArrayPlug = PlugDescriptor("vectorArray")
	inCache_ : InCachePlug = PlugDescriptor("inCache")

	# node attributes

	typeName = "cacheBlend"
	typeIdInt = 1129599563
	nodeLeafClassAttrs = ["end", "range", "start", "weight", "cacheData", "disableAll", "perPtWeights", "vectorArray", "inCache"]
	nodeLeafPlugs = ["cacheData", "disableAll", "inCache"]
	pass

