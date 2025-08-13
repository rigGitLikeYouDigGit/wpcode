

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
THsurfaceShape = retriever.getNodeCls("THsurfaceShape")
assert THsurfaceShape
if T.TYPE_CHECKING:
	from .. import THsurfaceShape

# add node doc



# region plug type defs
class StExpInPlug(Plug):
	parent : StDataInPlug = PlugDescriptor("stDataIn")
	node : StrataShape = None
	pass
class StMatrixInPlug(Plug):
	parent : StDataInPlug = PlugDescriptor("stDataIn")
	node : StrataShape = None
	pass
class StSpaceIndexInPlug(Plug):
	parent : StDataInPlug = PlugDescriptor("stDataIn")
	node : StrataShape = None
	pass
class StSpaceModeInPlug(Plug):
	parent : StDataInPlug = PlugDescriptor("stDataIn")
	node : StrataShape = None
	pass
class StSpaceNameInPlug(Plug):
	parent : StDataInPlug = PlugDescriptor("stDataIn")
	node : StrataShape = None
	pass
class StUVNInXPlug(Plug):
	parent : StUVNInPlug = PlugDescriptor("stUVNIn")
	node : StrataShape = None
	pass
class StUVNInYPlug(Plug):
	parent : StUVNInPlug = PlugDescriptor("stUVNIn")
	node : StrataShape = None
	pass
class StUVNInZPlug(Plug):
	parent : StUVNInPlug = PlugDescriptor("stUVNIn")
	node : StrataShape = None
	pass
class StUVNInPlug(Plug):
	parent : StDataInPlug = PlugDescriptor("stDataIn")
	stUVNInX_ : StUVNInXPlug = PlugDescriptor("stUVNInX")
	stUVNInx_ : StUVNInXPlug = PlugDescriptor("stUVNInX")
	stUVNInY_ : StUVNInYPlug = PlugDescriptor("stUVNInY")
	stUVNIny_ : StUVNInYPlug = PlugDescriptor("stUVNInY")
	stUVNInZ_ : StUVNInZPlug = PlugDescriptor("stUVNInZ")
	stUVNInz_ : StUVNInZPlug = PlugDescriptor("stUVNInZ")
	node : StrataShape = None
	pass
class StDataInPlug(Plug):
	stExpIn_ : StExpInPlug = PlugDescriptor("stExpIn")
	stExpIn_ : StExpInPlug = PlugDescriptor("stExpIn")
	stMatrixIn_ : StMatrixInPlug = PlugDescriptor("stMatrixIn")
	stMatrixIn_ : StMatrixInPlug = PlugDescriptor("stMatrixIn")
	stSpaceIndexIn_ : StSpaceIndexInPlug = PlugDescriptor("stSpaceIndexIn")
	stSpaceIndexIn_ : StSpaceIndexInPlug = PlugDescriptor("stSpaceIndexIn")
	stSpaceModeIn_ : StSpaceModeInPlug = PlugDescriptor("stSpaceModeIn")
	stSpaceModeIn_ : StSpaceModeInPlug = PlugDescriptor("stSpaceModeIn")
	stSpaceNameIn_ : StSpaceNameInPlug = PlugDescriptor("stSpaceNameIn")
	stSpaceNameIn_ : StSpaceNameInPlug = PlugDescriptor("stSpaceNameIn")
	stUVNIn_ : StUVNInPlug = PlugDescriptor("stUVNIn")
	stUVNIn_ : StUVNInPlug = PlugDescriptor("stUVNIn")
	node : StrataShape = None
	pass
class StCurveOutPlug(Plug):
	parent : StDataOutPlug = PlugDescriptor("stDataOut")
	node : StrataShape = None
	pass
class StExpOutPlug(Plug):
	parent : StDataOutPlug = PlugDescriptor("stDataOut")
	node : StrataShape = None
	pass
class StMatrixOutPlug(Plug):
	parent : StDataOutPlug = PlugDescriptor("stDataOut")
	node : StrataShape = None
	pass
class StDataOutPlug(Plug):
	stCurveOut_ : StCurveOutPlug = PlugDescriptor("stCurveOut")
	stCurveOut_ : StCurveOutPlug = PlugDescriptor("stCurveOut")
	stExpOut_ : StExpOutPlug = PlugDescriptor("stExpOut")
	stExpOut_ : StExpOutPlug = PlugDescriptor("stExpOut")
	stMatrixOut_ : StMatrixOutPlug = PlugDescriptor("stMatrixOut")
	stMatrixOut_ : StMatrixOutPlug = PlugDescriptor("stMatrixOut")
	node : StrataShape = None
	pass
class StInputPlug(Plug):
	node : StrataShape = None
	pass
class StOpNamePlug(Plug):
	node : StrataShape = None
	pass
class StOpNameOutPlug(Plug):
	node : StrataShape = None
	pass
class StOutputPlug(Plug):
	node : StrataShape = None
	pass
class StShowPointsPlug(Plug):
	node : StrataShape = None
	pass
# endregion


# define node class
class StrataShape(THsurfaceShape):
	stExpIn_ : StExpInPlug = PlugDescriptor("stExpIn")
	stMatrixIn_ : StMatrixInPlug = PlugDescriptor("stMatrixIn")
	stSpaceIndexIn_ : StSpaceIndexInPlug = PlugDescriptor("stSpaceIndexIn")
	stSpaceModeIn_ : StSpaceModeInPlug = PlugDescriptor("stSpaceModeIn")
	stSpaceNameIn_ : StSpaceNameInPlug = PlugDescriptor("stSpaceNameIn")
	stUVNInX_ : StUVNInXPlug = PlugDescriptor("stUVNInX")
	stUVNInY_ : StUVNInYPlug = PlugDescriptor("stUVNInY")
	stUVNInZ_ : StUVNInZPlug = PlugDescriptor("stUVNInZ")
	stUVNIn_ : StUVNInPlug = PlugDescriptor("stUVNIn")
	stDataIn_ : StDataInPlug = PlugDescriptor("stDataIn")
	stCurveOut_ : StCurveOutPlug = PlugDescriptor("stCurveOut")
	stExpOut_ : StExpOutPlug = PlugDescriptor("stExpOut")
	stMatrixOut_ : StMatrixOutPlug = PlugDescriptor("stMatrixOut")
	stDataOut_ : StDataOutPlug = PlugDescriptor("stDataOut")
	stInput_ : StInputPlug = PlugDescriptor("stInput")
	stOpName_ : StOpNamePlug = PlugDescriptor("stOpName")
	stOpNameOut_ : StOpNameOutPlug = PlugDescriptor("stOpNameOut")
	stOutput_ : StOutputPlug = PlugDescriptor("stOutput")
	stShowPoints_ : StShowPointsPlug = PlugDescriptor("stShowPoints")

	# node attributes

	typeName = "strataShape"
	typeIdInt = 1191075
	pass

