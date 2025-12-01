

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
	node : DisplacementShader = None
	pass
class DisplacementPlug(Plug):
	node : DisplacementShader = None
	pass
class DisplacementModePlug(Plug):
	node : DisplacementShader = None
	pass
class ScalePlug(Plug):
	node : DisplacementShader = None
	pass
class TangentXPlug(Plug):
	parent : TangentPlug = PlugDescriptor("tangent")
	node : DisplacementShader = None
	pass
class TangentYPlug(Plug):
	parent : TangentPlug = PlugDescriptor("tangent")
	node : DisplacementShader = None
	pass
class TangentZPlug(Plug):
	parent : TangentPlug = PlugDescriptor("tangent")
	node : DisplacementShader = None
	pass
class TangentPlug(Plug):
	tangentX_ : TangentXPlug = PlugDescriptor("tangentX")
	tx_ : TangentXPlug = PlugDescriptor("tangentX")
	tangentY_ : TangentYPlug = PlugDescriptor("tangentY")
	ty_ : TangentYPlug = PlugDescriptor("tangentY")
	tangentZ_ : TangentZPlug = PlugDescriptor("tangentZ")
	tz_ : TangentZPlug = PlugDescriptor("tangentZ")
	node : DisplacementShader = None
	pass
class VectorDisplacementXPlug(Plug):
	parent : VectorDisplacementPlug = PlugDescriptor("vectorDisplacement")
	node : DisplacementShader = None
	pass
class VectorDisplacementYPlug(Plug):
	parent : VectorDisplacementPlug = PlugDescriptor("vectorDisplacement")
	node : DisplacementShader = None
	pass
class VectorDisplacementZPlug(Plug):
	parent : VectorDisplacementPlug = PlugDescriptor("vectorDisplacement")
	node : DisplacementShader = None
	pass
class VectorDisplacementPlug(Plug):
	vectorDisplacementX_ : VectorDisplacementXPlug = PlugDescriptor("vectorDisplacementX")
	vdx_ : VectorDisplacementXPlug = PlugDescriptor("vectorDisplacementX")
	vectorDisplacementY_ : VectorDisplacementYPlug = PlugDescriptor("vectorDisplacementY")
	vdy_ : VectorDisplacementYPlug = PlugDescriptor("vectorDisplacementY")
	vectorDisplacementZ_ : VectorDisplacementZPlug = PlugDescriptor("vectorDisplacementZ")
	vdz_ : VectorDisplacementZPlug = PlugDescriptor("vectorDisplacementZ")
	node : DisplacementShader = None
	pass
class VectorEncodingPlug(Plug):
	node : DisplacementShader = None
	pass
class VectorSpacePlug(Plug):
	node : DisplacementShader = None
	pass
class YIsUpPlug(Plug):
	node : DisplacementShader = None
	pass
# endregion


# define node class
class DisplacementShader(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	displacement_ : DisplacementPlug = PlugDescriptor("displacement")
	displacementMode_ : DisplacementModePlug = PlugDescriptor("displacementMode")
	scale_ : ScalePlug = PlugDescriptor("scale")
	tangentX_ : TangentXPlug = PlugDescriptor("tangentX")
	tangentY_ : TangentYPlug = PlugDescriptor("tangentY")
	tangentZ_ : TangentZPlug = PlugDescriptor("tangentZ")
	tangent_ : TangentPlug = PlugDescriptor("tangent")
	vectorDisplacementX_ : VectorDisplacementXPlug = PlugDescriptor("vectorDisplacementX")
	vectorDisplacementY_ : VectorDisplacementYPlug = PlugDescriptor("vectorDisplacementY")
	vectorDisplacementZ_ : VectorDisplacementZPlug = PlugDescriptor("vectorDisplacementZ")
	vectorDisplacement_ : VectorDisplacementPlug = PlugDescriptor("vectorDisplacement")
	vectorEncoding_ : VectorEncodingPlug = PlugDescriptor("vectorEncoding")
	vectorSpace_ : VectorSpacePlug = PlugDescriptor("vectorSpace")
	yIsUp_ : YIsUpPlug = PlugDescriptor("yIsUp")

	# node attributes

	typeName = "displacementShader"
	apiTypeInt = 321
	apiTypeStr = "kDisplacementShader"
	typeIdInt = 1380209480
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "displacement", "displacementMode", "scale", "tangentX", "tangentY", "tangentZ", "tangent", "vectorDisplacementX", "vectorDisplacementY", "vectorDisplacementZ", "vectorDisplacement", "vectorEncoding", "vectorSpace", "yIsUp"]
	nodeLeafPlugs = ["binMembership", "displacement", "displacementMode", "scale", "tangent", "vectorDisplacement", "vectorEncoding", "vectorSpace", "yIsUp"]
	pass

