

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
	node : DataBlockTest = None
	pass
class Level3CCCS1Plug(Plug):
	parent : Level2CCCPlug = PlugDescriptor("level2CCC")
	node : DataBlockTest = None
	pass
class Level3CCCS2Plug(Plug):
	parent : Level2CCCPlug = PlugDescriptor("level2CCC")
	node : DataBlockTest = None
	pass
class Level2CCCPlug(Plug):
	parent : Level1CCPlug = PlugDescriptor("level1CC")
	level3CCCS1_ : Level3CCCS1Plug = PlugDescriptor("level3CCCS1")
	ccc1_ : Level3CCCS1Plug = PlugDescriptor("level3CCCS1")
	level3CCCS2_ : Level3CCCS2Plug = PlugDescriptor("level3CCCS2")
	ccc2_ : Level3CCCS2Plug = PlugDescriptor("level3CCCS2")
	node : DataBlockTest = None
	pass
class Level2CCMPlug(Plug):
	parent : Level1CCPlug = PlugDescriptor("level1CC")
	node : DataBlockTest = None
	pass
class Level3CCMCS1Plug(Plug):
	parent : Level2CCMCPlug = PlugDescriptor("level2CCMC")
	node : DataBlockTest = None
	pass
class Level3CCMCS2Plug(Plug):
	parent : Level2CCMCPlug = PlugDescriptor("level2CCMC")
	node : DataBlockTest = None
	pass
class Level2CCMCPlug(Plug):
	parent : Level1CCPlug = PlugDescriptor("level1CC")
	level3CCMCS1_ : Level3CCMCS1Plug = PlugDescriptor("level3CCMCS1")
	ccm1_ : Level3CCMCS1Plug = PlugDescriptor("level3CCMCS1")
	level3CCMCS2_ : Level3CCMCS2Plug = PlugDescriptor("level3CCMCS2")
	ccm2_ : Level3CCMCS2Plug = PlugDescriptor("level3CCMCS2")
	node : DataBlockTest = None
	pass
class Level2CCSPlug(Plug):
	parent : Level1CCPlug = PlugDescriptor("level1CC")
	node : DataBlockTest = None
	pass
class Level1CCPlug(Plug):
	parent : CompoundPlug = PlugDescriptor("compound")
	level2CCC_ : Level2CCCPlug = PlugDescriptor("level2CCC")
	ccc_ : Level2CCCPlug = PlugDescriptor("level2CCC")
	level2CCM_ : Level2CCMPlug = PlugDescriptor("level2CCM")
	ccm_ : Level2CCMPlug = PlugDescriptor("level2CCM")
	level2CCMC_ : Level2CCMCPlug = PlugDescriptor("level2CCMC")
	ccmc_ : Level2CCMCPlug = PlugDescriptor("level2CCMC")
	level2CCS_ : Level2CCSPlug = PlugDescriptor("level2CCS")
	ccs_ : Level2CCSPlug = PlugDescriptor("level2CCS")
	node : DataBlockTest = None
	pass
class Level1CMPlug(Plug):
	parent : CompoundPlug = PlugDescriptor("compound")
	node : DataBlockTest = None
	pass
class Level3CMCCS1Plug(Plug):
	parent : Level2CMCCPlug = PlugDescriptor("level2CMCC")
	node : DataBlockTest = None
	pass
class Level3CMCCS2Plug(Plug):
	parent : Level2CMCCPlug = PlugDescriptor("level2CMCC")
	node : DataBlockTest = None
	pass
class Level2CMCCPlug(Plug):
	parent : Level1CMCPlug = PlugDescriptor("level1CMC")
	level3CMCCS1_ : Level3CMCCS1Plug = PlugDescriptor("level3CMCCS1")
	cmc1_ : Level3CMCCS1Plug = PlugDescriptor("level3CMCCS1")
	level3CMCCS2_ : Level3CMCCS2Plug = PlugDescriptor("level3CMCCS2")
	cmc2_ : Level3CMCCS2Plug = PlugDescriptor("level3CMCCS2")
	node : DataBlockTest = None
	pass
class Level2CMCMPlug(Plug):
	parent : Level1CMCPlug = PlugDescriptor("level1CMC")
	node : DataBlockTest = None
	pass
class Level3CMCMCS1Plug(Plug):
	parent : Level2CMCMCPlug = PlugDescriptor("level2CMCMC")
	node : DataBlockTest = None
	pass
class Level3CMCMCS2Plug(Plug):
	parent : Level2CMCMCPlug = PlugDescriptor("level2CMCMC")
	node : DataBlockTest = None
	pass
class Level2CMCMCPlug(Plug):
	parent : Level1CMCPlug = PlugDescriptor("level1CMC")
	level3CMCMCS1_ : Level3CMCMCS1Plug = PlugDescriptor("level3CMCMCS1")
	cmm1_ : Level3CMCMCS1Plug = PlugDescriptor("level3CMCMCS1")
	level3CMCMCS2_ : Level3CMCMCS2Plug = PlugDescriptor("level3CMCMCS2")
	cmm2_ : Level3CMCMCS2Plug = PlugDescriptor("level3CMCMCS2")
	node : DataBlockTest = None
	pass
class Level2CMCSPlug(Plug):
	parent : Level1CMCPlug = PlugDescriptor("level1CMC")
	node : DataBlockTest = None
	pass
class Level1CMCPlug(Plug):
	parent : CompoundPlug = PlugDescriptor("compound")
	level2CMCC_ : Level2CMCCPlug = PlugDescriptor("level2CMCC")
	cmcc_ : Level2CMCCPlug = PlugDescriptor("level2CMCC")
	level2CMCM_ : Level2CMCMPlug = PlugDescriptor("level2CMCM")
	cmcm_ : Level2CMCMPlug = PlugDescriptor("level2CMCM")
	level2CMCMC_ : Level2CMCMCPlug = PlugDescriptor("level2CMCMC")
	cmmc_ : Level2CMCMCPlug = PlugDescriptor("level2CMCMC")
	level2CMCS_ : Level2CMCSPlug = PlugDescriptor("level2CMCS")
	cmcs_ : Level2CMCSPlug = PlugDescriptor("level2CMCS")
	node : DataBlockTest = None
	pass
class Level1CSPlug(Plug):
	parent : CompoundPlug = PlugDescriptor("compound")
	node : DataBlockTest = None
	pass
class CompoundPlug(Plug):
	level1CC_ : Level1CCPlug = PlugDescriptor("level1CC")
	l1cc_ : Level1CCPlug = PlugDescriptor("level1CC")
	level1CM_ : Level1CMPlug = PlugDescriptor("level1CM")
	cm_ : Level1CMPlug = PlugDescriptor("level1CM")
	level1CMC_ : Level1CMCPlug = PlugDescriptor("level1CMC")
	cmc_ : Level1CMCPlug = PlugDescriptor("level1CMC")
	level1CS_ : Level1CSPlug = PlugDescriptor("level1CS")
	cs_ : Level1CSPlug = PlugDescriptor("level1CS")
	node : DataBlockTest = None
	pass
class MultiPlug(Plug):
	node : DataBlockTest = None
	pass
class Level3MCCCS1Plug(Plug):
	parent : Level2MCCCPlug = PlugDescriptor("level2MCCC")
	node : DataBlockTest = None
	pass
class Level3MCCCS2Plug(Plug):
	parent : Level2MCCCPlug = PlugDescriptor("level2MCCC")
	node : DataBlockTest = None
	pass
class Level2MCCCPlug(Plug):
	parent : Level1MCCPlug = PlugDescriptor("level1MCC")
	level3MCCCS1_ : Level3MCCCS1Plug = PlugDescriptor("level3MCCCS1")
	mcc1_ : Level3MCCCS1Plug = PlugDescriptor("level3MCCCS1")
	level3MCCCS2_ : Level3MCCCS2Plug = PlugDescriptor("level3MCCCS2")
	mcc2_ : Level3MCCCS2Plug = PlugDescriptor("level3MCCCS2")
	node : DataBlockTest = None
	pass
class Level2MCCMPlug(Plug):
	parent : Level1MCCPlug = PlugDescriptor("level1MCC")
	node : DataBlockTest = None
	pass
class Level3MCCMCS1Plug(Plug):
	parent : Level2MCCMCPlug = PlugDescriptor("level2MCCMC")
	node : DataBlockTest = None
	pass
class Level3MCCMCS2Plug(Plug):
	parent : Level2MCCMCPlug = PlugDescriptor("level2MCCMC")
	node : DataBlockTest = None
	pass
class Level2MCCMCPlug(Plug):
	parent : Level1MCCPlug = PlugDescriptor("level1MCC")
	level3MCCMCS1_ : Level3MCCMCS1Plug = PlugDescriptor("level3MCCMCS1")
	mcm1_ : Level3MCCMCS1Plug = PlugDescriptor("level3MCCMCS1")
	level3MCCMCS2_ : Level3MCCMCS2Plug = PlugDescriptor("level3MCCMCS2")
	mcm2_ : Level3MCCMCS2Plug = PlugDescriptor("level3MCCMCS2")
	node : DataBlockTest = None
	pass
class Level2MCCSPlug(Plug):
	parent : Level1MCCPlug = PlugDescriptor("level1MCC")
	node : DataBlockTest = None
	pass
class Level1MCCPlug(Plug):
	parent : MultiCompoundPlug = PlugDescriptor("multiCompound")
	level2MCCC_ : Level2MCCCPlug = PlugDescriptor("level2MCCC")
	mccc_ : Level2MCCCPlug = PlugDescriptor("level2MCCC")
	level2MCCM_ : Level2MCCMPlug = PlugDescriptor("level2MCCM")
	mcm_ : Level2MCCMPlug = PlugDescriptor("level2MCCM")
	level2MCCMC_ : Level2MCCMCPlug = PlugDescriptor("level2MCCMC")
	mccm_ : Level2MCCMCPlug = PlugDescriptor("level2MCCMC")
	level2MCCS_ : Level2MCCSPlug = PlugDescriptor("level2MCCS")
	mccs_ : Level2MCCSPlug = PlugDescriptor("level2MCCS")
	node : DataBlockTest = None
	pass
class Level1MCMPlug(Plug):
	parent : MultiCompoundPlug = PlugDescriptor("multiCompound")
	node : DataBlockTest = None
	pass
class Level3MCMCCS1Plug(Plug):
	parent : Level2MCMCCPlug = PlugDescriptor("level2MCMCC")
	node : DataBlockTest = None
	pass
class Level3MCMCCS2Plug(Plug):
	parent : Level2MCMCCPlug = PlugDescriptor("level2MCMCC")
	node : DataBlockTest = None
	pass
class Level2MCMCCPlug(Plug):
	parent : Level1MCMCPlug = PlugDescriptor("level1MCMC")
	level3MCMCCS1_ : Level3MCMCCS1Plug = PlugDescriptor("level3MCMCCS1")
	mmc1_ : Level3MCMCCS1Plug = PlugDescriptor("level3MCMCCS1")
	level3MCMCCS2_ : Level3MCMCCS2Plug = PlugDescriptor("level3MCMCCS2")
	mmc2_ : Level3MCMCCS2Plug = PlugDescriptor("level3MCMCCS2")
	node : DataBlockTest = None
	pass
class Level2MCMCMPlug(Plug):
	parent : Level1MCMCPlug = PlugDescriptor("level1MCMC")
	node : DataBlockTest = None
	pass
class Level3MCMCMCS1Plug(Plug):
	parent : Level2MCMCMCPlug = PlugDescriptor("level2MCMCMC")
	node : DataBlockTest = None
	pass
class Level3MCMCMCS2Plug(Plug):
	parent : Level2MCMCMCPlug = PlugDescriptor("level2MCMCMC")
	node : DataBlockTest = None
	pass
class Level2MCMCMCPlug(Plug):
	parent : Level1MCMCPlug = PlugDescriptor("level1MCMC")
	level3MCMCMCS1_ : Level3MCMCMCS1Plug = PlugDescriptor("level3MCMCMCS1")
	mmm1_ : Level3MCMCMCS1Plug = PlugDescriptor("level3MCMCMCS1")
	level3MCMCMCS2_ : Level3MCMCMCS2Plug = PlugDescriptor("level3MCMCMCS2")
	mmm2_ : Level3MCMCMCS2Plug = PlugDescriptor("level3MCMCMCS2")
	node : DataBlockTest = None
	pass
class Level2MCMCSPlug(Plug):
	parent : Level1MCMCPlug = PlugDescriptor("level1MCMC")
	node : DataBlockTest = None
	pass
class Level1MCMCPlug(Plug):
	parent : MultiCompoundPlug = PlugDescriptor("multiCompound")
	level2MCMCC_ : Level2MCMCCPlug = PlugDescriptor("level2MCMCC")
	mmc_ : Level2MCMCCPlug = PlugDescriptor("level2MCMCC")
	level2MCMCM_ : Level2MCMCMPlug = PlugDescriptor("level2MCMCM")
	mmm_ : Level2MCMCMPlug = PlugDescriptor("level2MCMCM")
	level2MCMCMC_ : Level2MCMCMCPlug = PlugDescriptor("level2MCMCMC")
	mmmc_ : Level2MCMCMCPlug = PlugDescriptor("level2MCMCMC")
	level2MCMCS_ : Level2MCMCSPlug = PlugDescriptor("level2MCMCS")
	mms_ : Level2MCMCSPlug = PlugDescriptor("level2MCMCS")
	node : DataBlockTest = None
	pass
class Level1MCSPlug(Plug):
	parent : MultiCompoundPlug = PlugDescriptor("multiCompound")
	node : DataBlockTest = None
	pass
class MultiCompoundPlug(Plug):
	level1MCC_ : Level1MCCPlug = PlugDescriptor("level1MCC")
	mcc_ : Level1MCCPlug = PlugDescriptor("level1MCC")
	level1MCM_ : Level1MCMPlug = PlugDescriptor("level1MCM")
	mm_ : Level1MCMPlug = PlugDescriptor("level1MCM")
	level1MCMC_ : Level1MCMCPlug = PlugDescriptor("level1MCMC")
	mcmc_ : Level1MCMCPlug = PlugDescriptor("level1MCMC")
	level1MCS_ : Level1MCSPlug = PlugDescriptor("level1MCS")
	mcs_ : Level1MCSPlug = PlugDescriptor("level1MCS")
	node : DataBlockTest = None
	pass
class SinglePlug(Plug):
	node : DataBlockTest = None
	pass
# endregion


# define node class
class DataBlockTest(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	level3CCCS1_ : Level3CCCS1Plug = PlugDescriptor("level3CCCS1")
	level3CCCS2_ : Level3CCCS2Plug = PlugDescriptor("level3CCCS2")
	level2CCC_ : Level2CCCPlug = PlugDescriptor("level2CCC")
	level2CCM_ : Level2CCMPlug = PlugDescriptor("level2CCM")
	level3CCMCS1_ : Level3CCMCS1Plug = PlugDescriptor("level3CCMCS1")
	level3CCMCS2_ : Level3CCMCS2Plug = PlugDescriptor("level3CCMCS2")
	level2CCMC_ : Level2CCMCPlug = PlugDescriptor("level2CCMC")
	level2CCS_ : Level2CCSPlug = PlugDescriptor("level2CCS")
	level1CC_ : Level1CCPlug = PlugDescriptor("level1CC")
	level1CM_ : Level1CMPlug = PlugDescriptor("level1CM")
	level3CMCCS1_ : Level3CMCCS1Plug = PlugDescriptor("level3CMCCS1")
	level3CMCCS2_ : Level3CMCCS2Plug = PlugDescriptor("level3CMCCS2")
	level2CMCC_ : Level2CMCCPlug = PlugDescriptor("level2CMCC")
	level2CMCM_ : Level2CMCMPlug = PlugDescriptor("level2CMCM")
	level3CMCMCS1_ : Level3CMCMCS1Plug = PlugDescriptor("level3CMCMCS1")
	level3CMCMCS2_ : Level3CMCMCS2Plug = PlugDescriptor("level3CMCMCS2")
	level2CMCMC_ : Level2CMCMCPlug = PlugDescriptor("level2CMCMC")
	level2CMCS_ : Level2CMCSPlug = PlugDescriptor("level2CMCS")
	level1CMC_ : Level1CMCPlug = PlugDescriptor("level1CMC")
	level1CS_ : Level1CSPlug = PlugDescriptor("level1CS")
	compound_ : CompoundPlug = PlugDescriptor("compound")
	multi_ : MultiPlug = PlugDescriptor("multi")
	level3MCCCS1_ : Level3MCCCS1Plug = PlugDescriptor("level3MCCCS1")
	level3MCCCS2_ : Level3MCCCS2Plug = PlugDescriptor("level3MCCCS2")
	level2MCCC_ : Level2MCCCPlug = PlugDescriptor("level2MCCC")
	level2MCCM_ : Level2MCCMPlug = PlugDescriptor("level2MCCM")
	level3MCCMCS1_ : Level3MCCMCS1Plug = PlugDescriptor("level3MCCMCS1")
	level3MCCMCS2_ : Level3MCCMCS2Plug = PlugDescriptor("level3MCCMCS2")
	level2MCCMC_ : Level2MCCMCPlug = PlugDescriptor("level2MCCMC")
	level2MCCS_ : Level2MCCSPlug = PlugDescriptor("level2MCCS")
	level1MCC_ : Level1MCCPlug = PlugDescriptor("level1MCC")
	level1MCM_ : Level1MCMPlug = PlugDescriptor("level1MCM")
	level3MCMCCS1_ : Level3MCMCCS1Plug = PlugDescriptor("level3MCMCCS1")
	level3MCMCCS2_ : Level3MCMCCS2Plug = PlugDescriptor("level3MCMCCS2")
	level2MCMCC_ : Level2MCMCCPlug = PlugDescriptor("level2MCMCC")
	level2MCMCM_ : Level2MCMCMPlug = PlugDescriptor("level2MCMCM")
	level3MCMCMCS1_ : Level3MCMCMCS1Plug = PlugDescriptor("level3MCMCMCS1")
	level3MCMCMCS2_ : Level3MCMCMCS2Plug = PlugDescriptor("level3MCMCMCS2")
	level2MCMCMC_ : Level2MCMCMCPlug = PlugDescriptor("level2MCMCMC")
	level2MCMCS_ : Level2MCMCSPlug = PlugDescriptor("level2MCMCS")
	level1MCMC_ : Level1MCMCPlug = PlugDescriptor("level1MCMC")
	level1MCS_ : Level1MCSPlug = PlugDescriptor("level1MCS")
	multiCompound_ : MultiCompoundPlug = PlugDescriptor("multiCompound")
	single_ : SinglePlug = PlugDescriptor("single")

	# node attributes

	typeName = "dataBlockTest"
	typeIdInt = 1145197651
	pass

