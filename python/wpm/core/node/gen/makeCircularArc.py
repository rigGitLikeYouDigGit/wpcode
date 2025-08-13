

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class CenterXPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : MakeCircularArc = None
	pass
class CenterYPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : MakeCircularArc = None
	pass
class CenterZPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : MakeCircularArc = None
	pass
class CenterPlug(Plug):
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	cx_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	cy_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	cz_ : CenterZPlug = PlugDescriptor("centerZ")
	node : MakeCircularArc = None
	pass
class DegreePlug(Plug):
	node : MakeCircularArc = None
	pass
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : MakeCircularArc = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : MakeCircularArc = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : MakeCircularArc = None
	pass
class NormalPlug(Plug):
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nrx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	nry_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nrz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : MakeCircularArc = None
	pass
class OutputCurvePlug(Plug):
	node : MakeCircularArc = None
	pass
class SectionsPlug(Plug):
	node : MakeCircularArc = None
	pass
class SweepPlug(Plug):
	node : MakeCircularArc = None
	pass
# endregion


# define node class
class MakeCircularArc(AbstractBaseCreate):
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	center_ : CenterPlug = PlugDescriptor("center")
	degree_ : DegreePlug = PlugDescriptor("degree")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	sections_ : SectionsPlug = PlugDescriptor("sections")
	sweep_ : SweepPlug = PlugDescriptor("sweep")

	# node attributes

	typeName = "makeCircularArc"
	typeIdInt = 1313030482
	pass

