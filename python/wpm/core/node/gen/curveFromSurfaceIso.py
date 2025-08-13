

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
CurveFromSurface = retriever.getNodeCls("CurveFromSurface")
assert CurveFromSurface
if T.TYPE_CHECKING:
	from .. import CurveFromSurface

# add node doc



# region plug type defs
class IsoparmDirectionPlug(Plug):
	node : CurveFromSurfaceIso = None
	pass
class IsoparmValuePlug(Plug):
	node : CurveFromSurfaceIso = None
	pass
class RelativeValuePlug(Plug):
	node : CurveFromSurfaceIso = None
	pass
# endregion


# define node class
class CurveFromSurfaceIso(CurveFromSurface):
	isoparmDirection_ : IsoparmDirectionPlug = PlugDescriptor("isoparmDirection")
	isoparmValue_ : IsoparmValuePlug = PlugDescriptor("isoparmValue")
	relativeValue_ : RelativeValuePlug = PlugDescriptor("relativeValue")

	# node attributes

	typeName = "curveFromSurfaceIso"
	apiTypeInt = 61
	apiTypeStr = "kCurveFromSurfaceIso"
	typeIdInt = 1313035081
	MFnCls = om.MFnDependencyNode
	pass

