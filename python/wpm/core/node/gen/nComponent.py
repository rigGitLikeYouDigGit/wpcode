

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
	node : NComponent = None
	pass
class ComponentGroupIdPlug(Plug):
	node : NComponent = None
	pass
class ComponentIndicesPlug(Plug):
	node : NComponent = None
	pass
class ComponentTypePlug(Plug):
	node : NComponent = None
	pass
class ElementsPlug(Plug):
	node : NComponent = None
	pass
class GlueStrengthPlug(Plug):
	node : NComponent = None
	pass
class GlueStrengthMapPlug(Plug):
	node : NComponent = None
	pass
class GlueStrengthMapTypePlug(Plug):
	node : NComponent = None
	pass
class GlueStrengthPerVertexPlug(Plug):
	node : NComponent = None
	pass
class ObjectIdPlug(Plug):
	node : NComponent = None
	pass
class OutComponentPlug(Plug):
	node : NComponent = None
	pass
class StrengthPlug(Plug):
	node : NComponent = None
	pass
class StrengthMapPlug(Plug):
	node : NComponent = None
	pass
class StrengthMapTypePlug(Plug):
	node : NComponent = None
	pass
class StrengthPerVertexPlug(Plug):
	node : NComponent = None
	pass
class SurfacePlug(Plug):
	node : NComponent = None
	pass
class TangentStrengthPlug(Plug):
	node : NComponent = None
	pass
class WeightPlug(Plug):
	node : NComponent = None
	pass
class WeightMapPlug(Plug):
	node : NComponent = None
	pass
class WeightMapTypePlug(Plug):
	node : NComponent = None
	pass
class WeightPerVertexPlug(Plug):
	node : NComponent = None
	pass
# endregion


# define node class
class NComponent(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	componentGroupId_ : ComponentGroupIdPlug = PlugDescriptor("componentGroupId")
	componentIndices_ : ComponentIndicesPlug = PlugDescriptor("componentIndices")
	componentType_ : ComponentTypePlug = PlugDescriptor("componentType")
	elements_ : ElementsPlug = PlugDescriptor("elements")
	glueStrength_ : GlueStrengthPlug = PlugDescriptor("glueStrength")
	glueStrengthMap_ : GlueStrengthMapPlug = PlugDescriptor("glueStrengthMap")
	glueStrengthMapType_ : GlueStrengthMapTypePlug = PlugDescriptor("glueStrengthMapType")
	glueStrengthPerVertex_ : GlueStrengthPerVertexPlug = PlugDescriptor("glueStrengthPerVertex")
	objectId_ : ObjectIdPlug = PlugDescriptor("objectId")
	outComponent_ : OutComponentPlug = PlugDescriptor("outComponent")
	strength_ : StrengthPlug = PlugDescriptor("strength")
	strengthMap_ : StrengthMapPlug = PlugDescriptor("strengthMap")
	strengthMapType_ : StrengthMapTypePlug = PlugDescriptor("strengthMapType")
	strengthPerVertex_ : StrengthPerVertexPlug = PlugDescriptor("strengthPerVertex")
	surface_ : SurfacePlug = PlugDescriptor("surface")
	tangentStrength_ : TangentStrengthPlug = PlugDescriptor("tangentStrength")
	weight_ : WeightPlug = PlugDescriptor("weight")
	weightMap_ : WeightMapPlug = PlugDescriptor("weightMap")
	weightMapType_ : WeightMapTypePlug = PlugDescriptor("weightMapType")
	weightPerVertex_ : WeightPerVertexPlug = PlugDescriptor("weightPerVertex")

	# node attributes

	typeName = "nComponent"
	apiTypeInt = 994
	apiTypeStr = "kNComponent"
	typeIdInt = 1313033552
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "componentGroupId", "componentIndices", "componentType", "elements", "glueStrength", "glueStrengthMap", "glueStrengthMapType", "glueStrengthPerVertex", "objectId", "outComponent", "strength", "strengthMap", "strengthMapType", "strengthPerVertex", "surface", "tangentStrength", "weight", "weightMap", "weightMapType", "weightPerVertex"]
	nodeLeafPlugs = ["binMembership", "componentGroupId", "componentIndices", "componentType", "elements", "glueStrength", "glueStrengthMap", "glueStrengthMapType", "glueStrengthPerVertex", "objectId", "outComponent", "strength", "strengthMap", "strengthMapType", "strengthPerVertex", "surface", "tangentStrength", "weight", "weightMap", "weightMapType", "weightPerVertex"]
	pass

