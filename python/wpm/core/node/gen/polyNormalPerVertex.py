

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class NormalAddPlug(Plug):
	node : PolyNormalPerVertex = None
	pass
class NormalDeformPlug(Plug):
	node : PolyNormalPerVertex = None
	pass
class VertexNormalYPlug(Plug):
	parent : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	node : PolyNormalPerVertex = None
	pass
class VertexNormalZPlug(Plug):
	parent : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	node : PolyNormalPerVertex = None
	pass
class VertexNormalXYZPlug(Plug):
	parent : VertexNormalPlug = PlugDescriptor("vertexNormal")
	vertexNormalX_ : VertexNormalXPlug = PlugDescriptor("vertexNormalX")
	vxnx_ : VertexNormalXPlug = PlugDescriptor("vertexNormalX")
	vertexNormalY_ : VertexNormalYPlug = PlugDescriptor("vertexNormalY")
	vxny_ : VertexNormalYPlug = PlugDescriptor("vertexNormalY")
	vertexNormalZ_ : VertexNormalZPlug = PlugDescriptor("vertexNormalZ")
	vxnz_ : VertexNormalZPlug = PlugDescriptor("vertexNormalZ")
	node : PolyNormalPerVertex = None
	pass
class VertexNormalPlug(Plug):
	parent : NormalPerVertexPlug = PlugDescriptor("normalPerVertex")
	vertexFaceNormal_ : VertexFaceNormalPlug = PlugDescriptor("vertexFaceNormal")
	vfnl_ : VertexFaceNormalPlug = PlugDescriptor("vertexFaceNormal")
	vertexNormalXYZ_ : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	nxyz_ : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	node : PolyNormalPerVertex = None
	pass
class NormalPerVertexPlug(Plug):
	vertexNormal_ : VertexNormalPlug = PlugDescriptor("vertexNormal")
	vn_ : VertexNormalPlug = PlugDescriptor("vertexNormal")
	node : PolyNormalPerVertex = None
	pass
class VertexFaceNormalXPlug(Plug):
	parent : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	node : PolyNormalPerVertex = None
	pass
class VertexFaceNormalYPlug(Plug):
	parent : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	node : PolyNormalPerVertex = None
	pass
class VertexFaceNormalZPlug(Plug):
	parent : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	node : PolyNormalPerVertex = None
	pass
class VertexFaceNormalXYZPlug(Plug):
	parent : VertexFaceNormalPlug = PlugDescriptor("vertexFaceNormal")
	vertexFaceNormalX_ : VertexFaceNormalXPlug = PlugDescriptor("vertexFaceNormalX")
	vfnx_ : VertexFaceNormalXPlug = PlugDescriptor("vertexFaceNormalX")
	vertexFaceNormalY_ : VertexFaceNormalYPlug = PlugDescriptor("vertexFaceNormalY")
	vfny_ : VertexFaceNormalYPlug = PlugDescriptor("vertexFaceNormalY")
	vertexFaceNormalZ_ : VertexFaceNormalZPlug = PlugDescriptor("vertexFaceNormalZ")
	vfnz_ : VertexFaceNormalZPlug = PlugDescriptor("vertexFaceNormalZ")
	node : PolyNormalPerVertex = None
	pass
class VertexFaceNormalPlug(Plug):
	parent : VertexNormalPlug = PlugDescriptor("vertexNormal")
	vertexFaceNormalXYZ_ : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	fnxy_ : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	node : PolyNormalPerVertex = None
	pass
class VertexNormalXPlug(Plug):
	parent : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	node : PolyNormalPerVertex = None
	pass
# endregion


# define node class
class PolyNormalPerVertex(PolyModifier):
	normalAdd_ : NormalAddPlug = PlugDescriptor("normalAdd")
	normalDeform_ : NormalDeformPlug = PlugDescriptor("normalDeform")
	vertexNormalY_ : VertexNormalYPlug = PlugDescriptor("vertexNormalY")
	vertexNormalZ_ : VertexNormalZPlug = PlugDescriptor("vertexNormalZ")
	vertexNormalXYZ_ : VertexNormalXYZPlug = PlugDescriptor("vertexNormalXYZ")
	vertexNormal_ : VertexNormalPlug = PlugDescriptor("vertexNormal")
	normalPerVertex_ : NormalPerVertexPlug = PlugDescriptor("normalPerVertex")
	vertexFaceNormalX_ : VertexFaceNormalXPlug = PlugDescriptor("vertexFaceNormalX")
	vertexFaceNormalY_ : VertexFaceNormalYPlug = PlugDescriptor("vertexFaceNormalY")
	vertexFaceNormalZ_ : VertexFaceNormalZPlug = PlugDescriptor("vertexFaceNormalZ")
	vertexFaceNormalXYZ_ : VertexFaceNormalXYZPlug = PlugDescriptor("vertexFaceNormalXYZ")
	vertexFaceNormal_ : VertexFaceNormalPlug = PlugDescriptor("vertexFaceNormal")
	vertexNormalX_ : VertexNormalXPlug = PlugDescriptor("vertexNormalX")

	# node attributes

	typeName = "polyNormalPerVertex"
	apiTypeInt = 759
	apiTypeStr = "kPolyNormalPerVertex"
	typeIdInt = 1347309654
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["normalAdd", "normalDeform", "vertexNormalY", "vertexNormalZ", "vertexNormalXYZ", "vertexNormal", "normalPerVertex", "vertexFaceNormalX", "vertexFaceNormalY", "vertexFaceNormalZ", "vertexFaceNormalXYZ", "vertexFaceNormal", "vertexNormalX"]
	nodeLeafPlugs = ["normalAdd", "normalDeform", "normalPerVertex"]
	pass

