

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs
class BiasPlug(Plug):
	node : FilletCurve = None
	pass
class BlendControlPlug(Plug):
	node : FilletCurve = None
	pass
class CircularPlug(Plug):
	node : FilletCurve = None
	pass
class CurveParameter1Plug(Plug):
	node : FilletCurve = None
	pass
class CurveParameter2Plug(Plug):
	node : FilletCurve = None
	pass
class DepthPlug(Plug):
	node : FilletCurve = None
	pass
class DetachedCurve1Plug(Plug):
	node : FilletCurve = None
	pass
class DetachedCurve2Plug(Plug):
	node : FilletCurve = None
	pass
class FreeformBlendPlug(Plug):
	node : FilletCurve = None
	pass
class JoinPlug(Plug):
	node : FilletCurve = None
	pass
class OutputCurvePlug(Plug):
	node : FilletCurve = None
	pass
class PrimaryCurvePlug(Plug):
	node : FilletCurve = None
	pass
class RadiusPlug(Plug):
	node : FilletCurve = None
	pass
class SecondaryCurvePlug(Plug):
	node : FilletCurve = None
	pass
class TrimPlug(Plug):
	node : FilletCurve = None
	pass
# endregion


# define node class
class FilletCurve(AbstractBaseCreate):
	bias_ : BiasPlug = PlugDescriptor("bias")
	blendControl_ : BlendControlPlug = PlugDescriptor("blendControl")
	circular_ : CircularPlug = PlugDescriptor("circular")
	curveParameter1_ : CurveParameter1Plug = PlugDescriptor("curveParameter1")
	curveParameter2_ : CurveParameter2Plug = PlugDescriptor("curveParameter2")
	depth_ : DepthPlug = PlugDescriptor("depth")
	detachedCurve1_ : DetachedCurve1Plug = PlugDescriptor("detachedCurve1")
	detachedCurve2_ : DetachedCurve2Plug = PlugDescriptor("detachedCurve2")
	freeformBlend_ : FreeformBlendPlug = PlugDescriptor("freeformBlend")
	join_ : JoinPlug = PlugDescriptor("join")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	primaryCurve_ : PrimaryCurvePlug = PlugDescriptor("primaryCurve")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	secondaryCurve_ : SecondaryCurvePlug = PlugDescriptor("secondaryCurve")
	trim_ : TrimPlug = PlugDescriptor("trim")

	# node attributes

	typeName = "filletCurve"
	apiTypeInt = 70
	apiTypeStr = "kFilletCurve"
	typeIdInt = 1313227602
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["bias", "blendControl", "circular", "curveParameter1", "curveParameter2", "depth", "detachedCurve1", "detachedCurve2", "freeformBlend", "join", "outputCurve", "primaryCurve", "radius", "secondaryCurve", "trim"]
	nodeLeafPlugs = ["bias", "blendControl", "circular", "curveParameter1", "curveParameter2", "depth", "detachedCurve1", "detachedCurve2", "freeformBlend", "join", "outputCurve", "primaryCurve", "radius", "secondaryCurve", "trim"]
	pass

