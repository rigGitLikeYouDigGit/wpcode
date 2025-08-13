

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ControlPoint = retriever.getNodeCls("ControlPoint")
assert ControlPoint
if T.TYPE_CHECKING:
	from .. import ControlPoint

# add node doc



# region plug type defs
class FramesPlug(Plug):
	node : SnapshotShape = None
	pass
class PointsPlug(Plug):
	node : SnapshotShape = None
	pass
class ShowFramesPlug(Plug):
	node : SnapshotShape = None
	pass
# endregion


# define node class
class SnapshotShape(ControlPoint):
	frames_ : FramesPlug = PlugDescriptor("frames")
	points_ : PointsPlug = PlugDescriptor("points")
	showFrames_ : ShowFramesPlug = PlugDescriptor("showFrames")

	# node attributes

	typeName = "snapshotShape"
	typeIdInt = 1397966913
	pass

