

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ApplyRelOverride = Catalogue.ApplyRelOverride
else:
	from .. import retriever
	ApplyRelOverride = retriever.getNodeCls("ApplyRelOverride")
	assert ApplyRelOverride

# add node doc



# region plug type defs
class Multiply0Plug(Plug):
	parent : MultiplyPlug = PlugDescriptor("multiply")
	node : ApplyRel2FloatsOverride = None
	pass
class Multiply1Plug(Plug):
	parent : MultiplyPlug = PlugDescriptor("multiply")
	node : ApplyRel2FloatsOverride = None
	pass
class MultiplyPlug(Plug):
	multiply0_ : Multiply0Plug = PlugDescriptor("multiply0")
	mul0_ : Multiply0Plug = PlugDescriptor("multiply0")
	multiply1_ : Multiply1Plug = PlugDescriptor("multiply1")
	mul1_ : Multiply1Plug = PlugDescriptor("multiply1")
	node : ApplyRel2FloatsOverride = None
	pass
class Offset0Plug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : ApplyRel2FloatsOverride = None
	pass
class Offset1Plug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : ApplyRel2FloatsOverride = None
	pass
class OffsetPlug(Plug):
	offset0_ : Offset0Plug = PlugDescriptor("offset0")
	ofs0_ : Offset0Plug = PlugDescriptor("offset0")
	offset1_ : Offset1Plug = PlugDescriptor("offset1")
	ofs1_ : Offset1Plug = PlugDescriptor("offset1")
	node : ApplyRel2FloatsOverride = None
	pass
class Original0Plug(Plug):
	parent : OriginalPlug = PlugDescriptor("original")
	node : ApplyRel2FloatsOverride = None
	pass
class Original1Plug(Plug):
	parent : OriginalPlug = PlugDescriptor("original")
	node : ApplyRel2FloatsOverride = None
	pass
class OriginalPlug(Plug):
	original0_ : Original0Plug = PlugDescriptor("original0")
	ori0_ : Original0Plug = PlugDescriptor("original0")
	original1_ : Original1Plug = PlugDescriptor("original1")
	ori1_ : Original1Plug = PlugDescriptor("original1")
	node : ApplyRel2FloatsOverride = None
	pass
class Out0Plug(Plug):
	parent : OutPlug = PlugDescriptor("out")
	node : ApplyRel2FloatsOverride = None
	pass
class Out1Plug(Plug):
	parent : OutPlug = PlugDescriptor("out")
	node : ApplyRel2FloatsOverride = None
	pass
class OutPlug(Plug):
	out0_ : Out0Plug = PlugDescriptor("out0")
	o0_ : Out0Plug = PlugDescriptor("out0")
	out1_ : Out1Plug = PlugDescriptor("out1")
	o1_ : Out1Plug = PlugDescriptor("out1")
	node : ApplyRel2FloatsOverride = None
	pass
# endregion


# define node class
class ApplyRel2FloatsOverride(ApplyRelOverride):
	multiply0_ : Multiply0Plug = PlugDescriptor("multiply0")
	multiply1_ : Multiply1Plug = PlugDescriptor("multiply1")
	multiply_ : MultiplyPlug = PlugDescriptor("multiply")
	offset0_ : Offset0Plug = PlugDescriptor("offset0")
	offset1_ : Offset1Plug = PlugDescriptor("offset1")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	original0_ : Original0Plug = PlugDescriptor("original0")
	original1_ : Original1Plug = PlugDescriptor("original1")
	original_ : OriginalPlug = PlugDescriptor("original")
	out0_ : Out0Plug = PlugDescriptor("out0")
	out1_ : Out1Plug = PlugDescriptor("out1")
	out_ : OutPlug = PlugDescriptor("out")

	# node attributes

	typeName = "applyRel2FloatsOverride"
	typeIdInt = 1476395929
	nodeLeafClassAttrs = ["multiply0", "multiply1", "multiply", "offset0", "offset1", "offset", "original0", "original1", "original", "out0", "out1", "out"]
	nodeLeafPlugs = ["multiply", "offset", "original", "out"]
	pass

