

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
class FreezeNormalsPlug(Plug):
	node : TransformGeometry = None
	pass
class InputGeometryPlug(Plug):
	node : TransformGeometry = None
	pass
class InvertTransformPlug(Plug):
	node : TransformGeometry = None
	pass
class OutputGeometryPlug(Plug):
	node : TransformGeometry = None
	pass
class ReverseNormalsPlug(Plug):
	node : TransformGeometry = None
	pass
class TransformPlug(Plug):
	node : TransformGeometry = None
	pass
# endregion


# define node class
class TransformGeometry(AbstractBaseCreate):
	freezeNormals_ : FreezeNormalsPlug = PlugDescriptor("freezeNormals")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	invertTransform_ : InvertTransformPlug = PlugDescriptor("invertTransform")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")
	reverseNormals_ : ReverseNormalsPlug = PlugDescriptor("reverseNormals")
	transform_ : TransformPlug = PlugDescriptor("transform")

	# node attributes

	typeName = "transformGeometry"
	apiTypeInt = 609
	apiTypeStr = "kTransformGeometry"
	typeIdInt = 1413956943
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["freezeNormals", "inputGeometry", "invertTransform", "outputGeometry", "reverseNormals", "transform"]
	nodeLeafPlugs = ["freezeNormals", "inputGeometry", "invertTransform", "outputGeometry", "reverseNormals", "transform"]
	pass

