

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
	node : BlendMatrix = None
	pass
class EnablePlug(Plug):
	node : BlendMatrix = None
	pass
class EnvelopePlug(Plug):
	node : BlendMatrix = None
	pass
class InputMatrixPlug(Plug):
	node : BlendMatrix = None
	pass
class OutputMatrixPlug(Plug):
	node : BlendMatrix = None
	pass
class RotateWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : BlendMatrix = None
	pass
class ScaleWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : BlendMatrix = None
	pass
class ShearWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : BlendMatrix = None
	pass
class TargetMatrixPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : BlendMatrix = None
	pass
class TranslateWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : BlendMatrix = None
	pass
class UseMatrixPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : BlendMatrix = None
	pass
class WeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : BlendMatrix = None
	pass
class TargetPlug(Plug):
	rotateWeight_ : RotateWeightPlug = PlugDescriptor("rotateWeight")
	rot_ : RotateWeightPlug = PlugDescriptor("rotateWeight")
	scaleWeight_ : ScaleWeightPlug = PlugDescriptor("scaleWeight")
	sca_ : ScaleWeightPlug = PlugDescriptor("scaleWeight")
	shearWeight_ : ShearWeightPlug = PlugDescriptor("shearWeight")
	she_ : ShearWeightPlug = PlugDescriptor("shearWeight")
	targetMatrix_ : TargetMatrixPlug = PlugDescriptor("targetMatrix")
	tmat_ : TargetMatrixPlug = PlugDescriptor("targetMatrix")
	translateWeight_ : TranslateWeightPlug = PlugDescriptor("translateWeight")
	tra_ : TranslateWeightPlug = PlugDescriptor("translateWeight")
	useMatrix_ : UseMatrixPlug = PlugDescriptor("useMatrix")
	umt_ : UseMatrixPlug = PlugDescriptor("useMatrix")
	weight_ : WeightPlug = PlugDescriptor("weight")
	wgt_ : WeightPlug = PlugDescriptor("weight")
	node : BlendMatrix = None
	pass
# endregion


# define node class
class BlendMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	enable_ : EnablePlug = PlugDescriptor("enable")
	envelope_ : EnvelopePlug = PlugDescriptor("envelope")
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	outputMatrix_ : OutputMatrixPlug = PlugDescriptor("outputMatrix")
	rotateWeight_ : RotateWeightPlug = PlugDescriptor("rotateWeight")
	scaleWeight_ : ScaleWeightPlug = PlugDescriptor("scaleWeight")
	shearWeight_ : ShearWeightPlug = PlugDescriptor("shearWeight")
	targetMatrix_ : TargetMatrixPlug = PlugDescriptor("targetMatrix")
	translateWeight_ : TranslateWeightPlug = PlugDescriptor("translateWeight")
	useMatrix_ : UseMatrixPlug = PlugDescriptor("useMatrix")
	weight_ : WeightPlug = PlugDescriptor("weight")
	target_ : TargetPlug = PlugDescriptor("target")

	# node attributes

	typeName = "blendMatrix"
	apiTypeInt = 1137
	apiTypeStr = "kBlendMatrix"
	typeIdInt = 1112359252
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "enable", "envelope", "inputMatrix", "outputMatrix", "rotateWeight", "scaleWeight", "shearWeight", "targetMatrix", "translateWeight", "useMatrix", "weight", "target"]
	nodeLeafPlugs = ["binMembership", "enable", "envelope", "inputMatrix", "outputMatrix", "target"]
	pass

