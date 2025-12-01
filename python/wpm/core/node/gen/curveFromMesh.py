

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
	node : CurveFromMesh = None
	pass
class InputMeshPlug(Plug):
	node : CurveFromMesh = None
	pass
class OutputCurvePlug(Plug):
	node : CurveFromMesh = None
	pass
# endregion


# define node class
class CurveFromMesh(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inputMesh_ : InputMeshPlug = PlugDescriptor("inputMesh")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")

	# node attributes

	typeName = "curveFromMesh"
	typeIdInt = 1313031757
	nodeLeafClassAttrs = ["binMembership", "inputMesh", "outputCurve"]
	nodeLeafPlugs = ["binMembership", "inputMesh", "outputCurve"]
	pass

