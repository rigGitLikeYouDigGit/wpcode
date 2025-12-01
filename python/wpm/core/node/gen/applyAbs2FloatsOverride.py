

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ApplyAbsOverride = retriever.getNodeCls("ApplyAbsOverride")
assert ApplyAbsOverride
if T.TYPE_CHECKING:
	from .. import ApplyAbsOverride

# add node doc



# region plug type defs
class Original0Plug(Plug):
	parent : OriginalPlug = PlugDescriptor("original")
	node : ApplyAbs2FloatsOverride = None
	pass
class Original1Plug(Plug):
	parent : OriginalPlug = PlugDescriptor("original")
	node : ApplyAbs2FloatsOverride = None
	pass
class OriginalPlug(Plug):
	original0_ : Original0Plug = PlugDescriptor("original0")
	ori0_ : Original0Plug = PlugDescriptor("original0")
	original1_ : Original1Plug = PlugDescriptor("original1")
	ori1_ : Original1Plug = PlugDescriptor("original1")
	node : ApplyAbs2FloatsOverride = None
	pass
class Out0Plug(Plug):
	parent : OutPlug = PlugDescriptor("out")
	node : ApplyAbs2FloatsOverride = None
	pass
class Out1Plug(Plug):
	parent : OutPlug = PlugDescriptor("out")
	node : ApplyAbs2FloatsOverride = None
	pass
class OutPlug(Plug):
	out0_ : Out0Plug = PlugDescriptor("out0")
	o0_ : Out0Plug = PlugDescriptor("out0")
	out1_ : Out1Plug = PlugDescriptor("out1")
	o1_ : Out1Plug = PlugDescriptor("out1")
	node : ApplyAbs2FloatsOverride = None
	pass
class Value0Plug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : ApplyAbs2FloatsOverride = None
	pass
class Value1Plug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : ApplyAbs2FloatsOverride = None
	pass
class ValuePlug(Plug):
	value0_ : Value0Plug = PlugDescriptor("value0")
	val0_ : Value0Plug = PlugDescriptor("value0")
	value1_ : Value1Plug = PlugDescriptor("value1")
	val1_ : Value1Plug = PlugDescriptor("value1")
	node : ApplyAbs2FloatsOverride = None
	pass
# endregion


# define node class
class ApplyAbs2FloatsOverride(ApplyAbsOverride):
	original0_ : Original0Plug = PlugDescriptor("original0")
	original1_ : Original1Plug = PlugDescriptor("original1")
	original_ : OriginalPlug = PlugDescriptor("original")
	out0_ : Out0Plug = PlugDescriptor("out0")
	out1_ : Out1Plug = PlugDescriptor("out1")
	out_ : OutPlug = PlugDescriptor("out")
	value0_ : Value0Plug = PlugDescriptor("value0")
	value1_ : Value1Plug = PlugDescriptor("value1")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "applyAbs2FloatsOverride"
	typeIdInt = 1476395927
	nodeLeafClassAttrs = ["original0", "original1", "original", "out0", "out1", "out", "value0", "value1", "value"]
	nodeLeafPlugs = ["original", "out", "value"]
	pass

