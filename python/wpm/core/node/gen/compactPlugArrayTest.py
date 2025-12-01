

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
class AttrAMPlug(Plug):
	parent : AttrADPlug = PlugDescriptor("attrAD")
	node : CompactPlugArrayTest = None
	pass
class AttrAVPlug(Plug):
	parent : AttrADPlug = PlugDescriptor("attrAD")
	node : CompactPlugArrayTest = None
	pass
class AttrZFPlug(Plug):
	parent : AttrADPlug = PlugDescriptor("attrAD")
	node : CompactPlugArrayTest = None
	pass
class AttrADPlug(Plug):
	parent : AttrAAPlug = PlugDescriptor("attrAA")
	attrAM_ : AttrAMPlug = PlugDescriptor("attrAM")
	am_ : AttrAMPlug = PlugDescriptor("attrAM")
	attrAV_ : AttrAVPlug = PlugDescriptor("attrAV")
	av_ : AttrAVPlug = PlugDescriptor("attrAV")
	attrZF_ : AttrZFPlug = PlugDescriptor("attrZF")
	zf_ : AttrZFPlug = PlugDescriptor("attrZF")
	node : CompactPlugArrayTest = None
	pass
class AttrAPPlug(Plug):
	parent : AttrAGPlug = PlugDescriptor("attrAG")
	node : CompactPlugArrayTest = None
	pass
class AttrAYPlug(Plug):
	parent : AttrAGPlug = PlugDescriptor("attrAG")
	node : CompactPlugArrayTest = None
	pass
class AttrZIPlug(Plug):
	parent : AttrAGPlug = PlugDescriptor("attrAG")
	node : CompactPlugArrayTest = None
	pass
class AttrAGPlug(Plug):
	parent : AttrAAPlug = PlugDescriptor("attrAA")
	attrAP_ : AttrAPPlug = PlugDescriptor("attrAP")
	ap_ : AttrAPPlug = PlugDescriptor("attrAP")
	attrAY_ : AttrAYPlug = PlugDescriptor("attrAY")
	ay_ : AttrAYPlug = PlugDescriptor("attrAY")
	attrZI_ : AttrZIPlug = PlugDescriptor("attrZI")
	zi_ : AttrZIPlug = PlugDescriptor("attrZI")
	node : CompactPlugArrayTest = None
	pass
class AttrASPlug(Plug):
	parent : AttrAJPlug = PlugDescriptor("attrAJ")
	node : CompactPlugArrayTest = None
	pass
class AttrZCPlug(Plug):
	parent : AttrAJPlug = PlugDescriptor("attrAJ")
	node : CompactPlugArrayTest = None
	pass
class AttrZLPlug(Plug):
	parent : AttrAJPlug = PlugDescriptor("attrAJ")
	node : CompactPlugArrayTest = None
	pass
class AttrAJPlug(Plug):
	parent : AttrAAPlug = PlugDescriptor("attrAA")
	attrAS_ : AttrASPlug = PlugDescriptor("attrAS")
	as_ : AttrASPlug = PlugDescriptor("attrAS")
	attrZC_ : AttrZCPlug = PlugDescriptor("attrZC")
	zc_ : AttrZCPlug = PlugDescriptor("attrZC")
	attrZL_ : AttrZLPlug = PlugDescriptor("attrZL")
	zl_ : AttrZLPlug = PlugDescriptor("attrZL")
	node : CompactPlugArrayTest = None
	pass
class AttrAAPlug(Plug):
	attrAD_ : AttrADPlug = PlugDescriptor("attrAD")
	ad_ : AttrADPlug = PlugDescriptor("attrAD")
	attrAG_ : AttrAGPlug = PlugDescriptor("attrAG")
	ag_ : AttrAGPlug = PlugDescriptor("attrAG")
	attrAJ_ : AttrAJPlug = PlugDescriptor("attrAJ")
	aj_ : AttrAJPlug = PlugDescriptor("attrAJ")
	node : CompactPlugArrayTest = None
	pass
class AttrANPlug(Plug):
	parent : AttrAEPlug = PlugDescriptor("attrAE")
	node : CompactPlugArrayTest = None
	pass
class AttrAWPlug(Plug):
	parent : AttrAEPlug = PlugDescriptor("attrAE")
	node : CompactPlugArrayTest = None
	pass
class AttrZGPlug(Plug):
	parent : AttrAEPlug = PlugDescriptor("attrAE")
	node : CompactPlugArrayTest = None
	pass
class AttrAEPlug(Plug):
	parent : AttrABPlug = PlugDescriptor("attrAB")
	attrAN_ : AttrANPlug = PlugDescriptor("attrAN")
	an_ : AttrANPlug = PlugDescriptor("attrAN")
	attrAW_ : AttrAWPlug = PlugDescriptor("attrAW")
	aw_ : AttrAWPlug = PlugDescriptor("attrAW")
	attrZG_ : AttrZGPlug = PlugDescriptor("attrZG")
	zg_ : AttrZGPlug = PlugDescriptor("attrZG")
	node : CompactPlugArrayTest = None
	pass
class AttrAQPlug(Plug):
	parent : AttrAHPlug = PlugDescriptor("attrAH")
	node : CompactPlugArrayTest = None
	pass
class AttrAZPlug(Plug):
	parent : AttrAHPlug = PlugDescriptor("attrAH")
	node : CompactPlugArrayTest = None
	pass
class AttrZJPlug(Plug):
	parent : AttrAHPlug = PlugDescriptor("attrAH")
	node : CompactPlugArrayTest = None
	pass
class AttrAHPlug(Plug):
	parent : AttrABPlug = PlugDescriptor("attrAB")
	attrAQ_ : AttrAQPlug = PlugDescriptor("attrAQ")
	aq_ : AttrAQPlug = PlugDescriptor("attrAQ")
	attrAZ_ : AttrAZPlug = PlugDescriptor("attrAZ")
	az_ : AttrAZPlug = PlugDescriptor("attrAZ")
	attrZJ_ : AttrZJPlug = PlugDescriptor("attrZJ")
	zj_ : AttrZJPlug = PlugDescriptor("attrZJ")
	node : CompactPlugArrayTest = None
	pass
class AttrATPlug(Plug):
	parent : AttrAKPlug = PlugDescriptor("attrAK")
	node : CompactPlugArrayTest = None
	pass
class AttrZDPlug(Plug):
	parent : AttrAKPlug = PlugDescriptor("attrAK")
	node : CompactPlugArrayTest = None
	pass
class AttrZMPlug(Plug):
	parent : AttrAKPlug = PlugDescriptor("attrAK")
	node : CompactPlugArrayTest = None
	pass
class AttrAKPlug(Plug):
	parent : AttrABPlug = PlugDescriptor("attrAB")
	attrAT_ : AttrATPlug = PlugDescriptor("attrAT")
	at_ : AttrATPlug = PlugDescriptor("attrAT")
	attrZD_ : AttrZDPlug = PlugDescriptor("attrZD")
	zd_ : AttrZDPlug = PlugDescriptor("attrZD")
	attrZM_ : AttrZMPlug = PlugDescriptor("attrZM")
	zm_ : AttrZMPlug = PlugDescriptor("attrZM")
	node : CompactPlugArrayTest = None
	pass
class AttrABPlug(Plug):
	attrAE_ : AttrAEPlug = PlugDescriptor("attrAE")
	ae_ : AttrAEPlug = PlugDescriptor("attrAE")
	attrAH_ : AttrAHPlug = PlugDescriptor("attrAH")
	ah_ : AttrAHPlug = PlugDescriptor("attrAH")
	attrAK_ : AttrAKPlug = PlugDescriptor("attrAK")
	ak_ : AttrAKPlug = PlugDescriptor("attrAK")
	node : CompactPlugArrayTest = None
	pass
class AttrAOPlug(Plug):
	parent : AttrAFPlug = PlugDescriptor("attrAF")
	node : CompactPlugArrayTest = None
	pass
class AttrAXPlug(Plug):
	parent : AttrAFPlug = PlugDescriptor("attrAF")
	node : CompactPlugArrayTest = None
	pass
class AttrZHPlug(Plug):
	parent : AttrAFPlug = PlugDescriptor("attrAF")
	node : CompactPlugArrayTest = None
	pass
class AttrAFPlug(Plug):
	parent : AttrACPlug = PlugDescriptor("attrAC")
	attrAO_ : AttrAOPlug = PlugDescriptor("attrAO")
	ao_ : AttrAOPlug = PlugDescriptor("attrAO")
	attrAX_ : AttrAXPlug = PlugDescriptor("attrAX")
	ax_ : AttrAXPlug = PlugDescriptor("attrAX")
	attrZH_ : AttrZHPlug = PlugDescriptor("attrZH")
	zh_ : AttrZHPlug = PlugDescriptor("attrZH")
	node : CompactPlugArrayTest = None
	pass
class AttrARPlug(Plug):
	parent : AttrAIPlug = PlugDescriptor("attrAI")
	node : CompactPlugArrayTest = None
	pass
class AttrZBPlug(Plug):
	parent : AttrAIPlug = PlugDescriptor("attrAI")
	node : CompactPlugArrayTest = None
	pass
class AttrZKPlug(Plug):
	parent : AttrAIPlug = PlugDescriptor("attrAI")
	node : CompactPlugArrayTest = None
	pass
class AttrAIPlug(Plug):
	parent : AttrACPlug = PlugDescriptor("attrAC")
	attrAR_ : AttrARPlug = PlugDescriptor("attrAR")
	ar_ : AttrARPlug = PlugDescriptor("attrAR")
	attrZB_ : AttrZBPlug = PlugDescriptor("attrZB")
	zb_ : AttrZBPlug = PlugDescriptor("attrZB")
	attrZK_ : AttrZKPlug = PlugDescriptor("attrZK")
	zk_ : AttrZKPlug = PlugDescriptor("attrZK")
	node : CompactPlugArrayTest = None
	pass
class AttrAUPlug(Plug):
	parent : AttrALPlug = PlugDescriptor("attrAL")
	node : CompactPlugArrayTest = None
	pass
class AttrZEPlug(Plug):
	parent : AttrALPlug = PlugDescriptor("attrAL")
	node : CompactPlugArrayTest = None
	pass
class AttrZNPlug(Plug):
	parent : AttrALPlug = PlugDescriptor("attrAL")
	node : CompactPlugArrayTest = None
	pass
class AttrALPlug(Plug):
	parent : AttrACPlug = PlugDescriptor("attrAC")
	attrAU_ : AttrAUPlug = PlugDescriptor("attrAU")
	au_ : AttrAUPlug = PlugDescriptor("attrAU")
	attrZE_ : AttrZEPlug = PlugDescriptor("attrZE")
	ze_ : AttrZEPlug = PlugDescriptor("attrZE")
	attrZN_ : AttrZNPlug = PlugDescriptor("attrZN")
	zn_ : AttrZNPlug = PlugDescriptor("attrZN")
	node : CompactPlugArrayTest = None
	pass
class AttrACPlug(Plug):
	attrAF_ : AttrAFPlug = PlugDescriptor("attrAF")
	af_ : AttrAFPlug = PlugDescriptor("attrAF")
	attrAI_ : AttrAIPlug = PlugDescriptor("attrAI")
	ai_ : AttrAIPlug = PlugDescriptor("attrAI")
	attrAL_ : AttrALPlug = PlugDescriptor("attrAL")
	al_ : AttrALPlug = PlugDescriptor("attrAL")
	node : CompactPlugArrayTest = None
	pass
class BinMembershipPlug(Plug):
	node : CompactPlugArrayTest = None
	pass
# endregion


# define node class
class CompactPlugArrayTest(_BASE_):
	attrAM_ : AttrAMPlug = PlugDescriptor("attrAM")
	attrAV_ : AttrAVPlug = PlugDescriptor("attrAV")
	attrZF_ : AttrZFPlug = PlugDescriptor("attrZF")
	attrAD_ : AttrADPlug = PlugDescriptor("attrAD")
	attrAP_ : AttrAPPlug = PlugDescriptor("attrAP")
	attrAY_ : AttrAYPlug = PlugDescriptor("attrAY")
	attrZI_ : AttrZIPlug = PlugDescriptor("attrZI")
	attrAG_ : AttrAGPlug = PlugDescriptor("attrAG")
	attrAS_ : AttrASPlug = PlugDescriptor("attrAS")
	attrZC_ : AttrZCPlug = PlugDescriptor("attrZC")
	attrZL_ : AttrZLPlug = PlugDescriptor("attrZL")
	attrAJ_ : AttrAJPlug = PlugDescriptor("attrAJ")
	attrAA_ : AttrAAPlug = PlugDescriptor("attrAA")
	attrAN_ : AttrANPlug = PlugDescriptor("attrAN")
	attrAW_ : AttrAWPlug = PlugDescriptor("attrAW")
	attrZG_ : AttrZGPlug = PlugDescriptor("attrZG")
	attrAE_ : AttrAEPlug = PlugDescriptor("attrAE")
	attrAQ_ : AttrAQPlug = PlugDescriptor("attrAQ")
	attrAZ_ : AttrAZPlug = PlugDescriptor("attrAZ")
	attrZJ_ : AttrZJPlug = PlugDescriptor("attrZJ")
	attrAH_ : AttrAHPlug = PlugDescriptor("attrAH")
	attrAT_ : AttrATPlug = PlugDescriptor("attrAT")
	attrZD_ : AttrZDPlug = PlugDescriptor("attrZD")
	attrZM_ : AttrZMPlug = PlugDescriptor("attrZM")
	attrAK_ : AttrAKPlug = PlugDescriptor("attrAK")
	attrAB_ : AttrABPlug = PlugDescriptor("attrAB")
	attrAO_ : AttrAOPlug = PlugDescriptor("attrAO")
	attrAX_ : AttrAXPlug = PlugDescriptor("attrAX")
	attrZH_ : AttrZHPlug = PlugDescriptor("attrZH")
	attrAF_ : AttrAFPlug = PlugDescriptor("attrAF")
	attrAR_ : AttrARPlug = PlugDescriptor("attrAR")
	attrZB_ : AttrZBPlug = PlugDescriptor("attrZB")
	attrZK_ : AttrZKPlug = PlugDescriptor("attrZK")
	attrAI_ : AttrAIPlug = PlugDescriptor("attrAI")
	attrAU_ : AttrAUPlug = PlugDescriptor("attrAU")
	attrZE_ : AttrZEPlug = PlugDescriptor("attrZE")
	attrZN_ : AttrZNPlug = PlugDescriptor("attrZN")
	attrAL_ : AttrALPlug = PlugDescriptor("attrAL")
	attrAC_ : AttrACPlug = PlugDescriptor("attrAC")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")

	# node attributes

	typeName = "compactPlugArrayTest"
	typeIdInt = 1129333076
	nodeLeafClassAttrs = ["attrAM", "attrAV", "attrZF", "attrAD", "attrAP", "attrAY", "attrZI", "attrAG", "attrAS", "attrZC", "attrZL", "attrAJ", "attrAA", "attrAN", "attrAW", "attrZG", "attrAE", "attrAQ", "attrAZ", "attrZJ", "attrAH", "attrAT", "attrZD", "attrZM", "attrAK", "attrAB", "attrAO", "attrAX", "attrZH", "attrAF", "attrAR", "attrZB", "attrZK", "attrAI", "attrAU", "attrZE", "attrZN", "attrAL", "attrAC", "binMembership"]
	nodeLeafPlugs = ["attrAA", "attrAB", "attrAC", "binMembership"]
	pass

