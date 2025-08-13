

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
	node : ApplyAbs3FloatsOverride = None
	pass
class Original1Plug(Plug):
	parent : OriginalPlug = PlugDescriptor("original")
	node : ApplyAbs3FloatsOverride = None
	pass
class Original2Plug(Plug):
	parent : OriginalPlug = PlugDescriptor("original")
	node : ApplyAbs3FloatsOverride = None
	pass
class OriginalPlug(Plug):
	original0_ : Original0Plug = PlugDescriptor("original0")
	ori0_ : Original0Plug = PlugDescriptor("original0")
	original1_ : Original1Plug = PlugDescriptor("original1")
	ori1_ : Original1Plug = PlugDescriptor("original1")
	original2_ : Original2Plug = PlugDescriptor("original2")
	ori2_ : Original2Plug = PlugDescriptor("original2")
	node : ApplyAbs3FloatsOverride = None
	pass
class Out0Plug(Plug):
	parent : OutPlug = PlugDescriptor("out")
	node : ApplyAbs3FloatsOverride = None
	pass
class Out1Plug(Plug):
	parent : OutPlug = PlugDescriptor("out")
	node : ApplyAbs3FloatsOverride = None
	pass
class Out2Plug(Plug):
	parent : OutPlug = PlugDescriptor("out")
	node : ApplyAbs3FloatsOverride = None
	pass
class OutPlug(Plug):
	out0_ : Out0Plug = PlugDescriptor("out0")
	o0_ : Out0Plug = PlugDescriptor("out0")
	out1_ : Out1Plug = PlugDescriptor("out1")
	o1_ : Out1Plug = PlugDescriptor("out1")
	out2_ : Out2Plug = PlugDescriptor("out2")
	o2_ : Out2Plug = PlugDescriptor("out2")
	node : ApplyAbs3FloatsOverride = None
	pass
class Value0Plug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : ApplyAbs3FloatsOverride = None
	pass
class Value1Plug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : ApplyAbs3FloatsOverride = None
	pass
class Value2Plug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : ApplyAbs3FloatsOverride = None
	pass
class ValuePlug(Plug):
	value0_ : Value0Plug = PlugDescriptor("value0")
	val0_ : Value0Plug = PlugDescriptor("value0")
	value1_ : Value1Plug = PlugDescriptor("value1")
	val1_ : Value1Plug = PlugDescriptor("value1")
	value2_ : Value2Plug = PlugDescriptor("value2")
	val2_ : Value2Plug = PlugDescriptor("value2")
	node : ApplyAbs3FloatsOverride = None
	pass
# endregion


# define node class
class ApplyAbs3FloatsOverride(ApplyAbsOverride):
	original0_ : Original0Plug = PlugDescriptor("original0")
	original1_ : Original1Plug = PlugDescriptor("original1")
	original2_ : Original2Plug = PlugDescriptor("original2")
	original_ : OriginalPlug = PlugDescriptor("original")
	out0_ : Out0Plug = PlugDescriptor("out0")
	out1_ : Out1Plug = PlugDescriptor("out1")
	out2_ : Out2Plug = PlugDescriptor("out2")
	out_ : OutPlug = PlugDescriptor("out")
	value0_ : Value0Plug = PlugDescriptor("value0")
	value1_ : Value1Plug = PlugDescriptor("value1")
	value2_ : Value2Plug = PlugDescriptor("value2")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "applyAbs3FloatsOverride"
	typeIdInt = 1476395905
	pass

