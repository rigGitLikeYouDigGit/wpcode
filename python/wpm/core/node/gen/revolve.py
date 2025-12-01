

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
class AutoCorrectNormalPlug(Plug):
	node : Revolve = None
	pass
class AxisXPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : Revolve = None
	pass
class AxisYPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : Revolve = None
	pass
class AxisZPlug(Plug):
	parent : AxisPlug = PlugDescriptor("axis")
	node : Revolve = None
	pass
class AxisPlug(Plug):
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axx_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axy_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axz_ : AxisZPlug = PlugDescriptor("axisZ")
	node : Revolve = None
	pass
class AxisChoicePlug(Plug):
	node : Revolve = None
	pass
class BridgePlug(Plug):
	node : Revolve = None
	pass
class BridgeCurvePlug(Plug):
	node : Revolve = None
	pass
class CompAnchorXPlug(Plug):
	parent : CompAnchorPlug = PlugDescriptor("compAnchor")
	node : Revolve = None
	pass
class CompAnchorYPlug(Plug):
	parent : CompAnchorPlug = PlugDescriptor("compAnchor")
	node : Revolve = None
	pass
class CompAnchorZPlug(Plug):
	parent : CompAnchorPlug = PlugDescriptor("compAnchor")
	node : Revolve = None
	pass
class CompAnchorPlug(Plug):
	compAnchorX_ : CompAnchorXPlug = PlugDescriptor("compAnchorX")
	cnx_ : CompAnchorXPlug = PlugDescriptor("compAnchorX")
	compAnchorY_ : CompAnchorYPlug = PlugDescriptor("compAnchorY")
	cny_ : CompAnchorYPlug = PlugDescriptor("compAnchorY")
	compAnchorZ_ : CompAnchorZPlug = PlugDescriptor("compAnchorZ")
	cnz_ : CompAnchorZPlug = PlugDescriptor("compAnchorZ")
	node : Revolve = None
	pass
class CompAxisXPlug(Plug):
	parent : CompAxisPlug = PlugDescriptor("compAxis")
	node : Revolve = None
	pass
class CompAxisYPlug(Plug):
	parent : CompAxisPlug = PlugDescriptor("compAxis")
	node : Revolve = None
	pass
class CompAxisZPlug(Plug):
	parent : CompAxisPlug = PlugDescriptor("compAxis")
	node : Revolve = None
	pass
class CompAxisPlug(Plug):
	compAxisX_ : CompAxisXPlug = PlugDescriptor("compAxisX")
	cax_ : CompAxisXPlug = PlugDescriptor("compAxisX")
	compAxisY_ : CompAxisYPlug = PlugDescriptor("compAxisY")
	cay_ : CompAxisYPlug = PlugDescriptor("compAxisY")
	compAxisZ_ : CompAxisZPlug = PlugDescriptor("compAxisZ")
	caz_ : CompAxisZPlug = PlugDescriptor("compAxisZ")
	node : Revolve = None
	pass
class CompAxisChoicePlug(Plug):
	node : Revolve = None
	pass
class CompPivotXPlug(Plug):
	parent : CompPivotPlug = PlugDescriptor("compPivot")
	node : Revolve = None
	pass
class CompPivotYPlug(Plug):
	parent : CompPivotPlug = PlugDescriptor("compPivot")
	node : Revolve = None
	pass
class CompPivotZPlug(Plug):
	parent : CompPivotPlug = PlugDescriptor("compPivot")
	node : Revolve = None
	pass
class CompPivotPlug(Plug):
	compPivotX_ : CompPivotXPlug = PlugDescriptor("compPivotX")
	cpx_ : CompPivotXPlug = PlugDescriptor("compPivotX")
	compPivotY_ : CompPivotYPlug = PlugDescriptor("compPivotY")
	cpy_ : CompPivotYPlug = PlugDescriptor("compPivotY")
	compPivotZ_ : CompPivotZPlug = PlugDescriptor("compPivotZ")
	cpz_ : CompPivotZPlug = PlugDescriptor("compPivotZ")
	node : Revolve = None
	pass
class ComputePivotAndAxisPlug(Plug):
	node : Revolve = None
	pass
class DegreePlug(Plug):
	node : Revolve = None
	pass
class EndSweepPlug(Plug):
	node : Revolve = None
	pass
class InputCurvePlug(Plug):
	node : Revolve = None
	pass
class OutputSurfacePlug(Plug):
	node : Revolve = None
	pass
class PivotXPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : Revolve = None
	pass
class PivotYPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : Revolve = None
	pass
class PivotZPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : Revolve = None
	pass
class PivotPlug(Plug):
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	px_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	py_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pz_ : PivotZPlug = PlugDescriptor("pivotZ")
	node : Revolve = None
	pass
class RadiusPlug(Plug):
	node : Revolve = None
	pass
class RadiusAnchorPlug(Plug):
	node : Revolve = None
	pass
class SectionsPlug(Plug):
	node : Revolve = None
	pass
class StartSweepPlug(Plug):
	node : Revolve = None
	pass
class TolerancePlug(Plug):
	node : Revolve = None
	pass
class UseTolerancePlug(Plug):
	node : Revolve = None
	pass
# endregion


# define node class
class Revolve(AbstractBaseCreate):
	autoCorrectNormal_ : AutoCorrectNormalPlug = PlugDescriptor("autoCorrectNormal")
	axisX_ : AxisXPlug = PlugDescriptor("axisX")
	axisY_ : AxisYPlug = PlugDescriptor("axisY")
	axisZ_ : AxisZPlug = PlugDescriptor("axisZ")
	axis_ : AxisPlug = PlugDescriptor("axis")
	axisChoice_ : AxisChoicePlug = PlugDescriptor("axisChoice")
	bridge_ : BridgePlug = PlugDescriptor("bridge")
	bridgeCurve_ : BridgeCurvePlug = PlugDescriptor("bridgeCurve")
	compAnchorX_ : CompAnchorXPlug = PlugDescriptor("compAnchorX")
	compAnchorY_ : CompAnchorYPlug = PlugDescriptor("compAnchorY")
	compAnchorZ_ : CompAnchorZPlug = PlugDescriptor("compAnchorZ")
	compAnchor_ : CompAnchorPlug = PlugDescriptor("compAnchor")
	compAxisX_ : CompAxisXPlug = PlugDescriptor("compAxisX")
	compAxisY_ : CompAxisYPlug = PlugDescriptor("compAxisY")
	compAxisZ_ : CompAxisZPlug = PlugDescriptor("compAxisZ")
	compAxis_ : CompAxisPlug = PlugDescriptor("compAxis")
	compAxisChoice_ : CompAxisChoicePlug = PlugDescriptor("compAxisChoice")
	compPivotX_ : CompPivotXPlug = PlugDescriptor("compPivotX")
	compPivotY_ : CompPivotYPlug = PlugDescriptor("compPivotY")
	compPivotZ_ : CompPivotZPlug = PlugDescriptor("compPivotZ")
	compPivot_ : CompPivotPlug = PlugDescriptor("compPivot")
	computePivotAndAxis_ : ComputePivotAndAxisPlug = PlugDescriptor("computePivotAndAxis")
	degree_ : DegreePlug = PlugDescriptor("degree")
	endSweep_ : EndSweepPlug = PlugDescriptor("endSweep")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pivot_ : PivotPlug = PlugDescriptor("pivot")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	radiusAnchor_ : RadiusAnchorPlug = PlugDescriptor("radiusAnchor")
	sections_ : SectionsPlug = PlugDescriptor("sections")
	startSweep_ : StartSweepPlug = PlugDescriptor("startSweep")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")
	useTolerance_ : UseTolerancePlug = PlugDescriptor("useTolerance")

	# node attributes

	typeName = "revolve"
	apiTypeInt = 94
	apiTypeStr = "kRevolve"
	typeIdInt = 1314018892
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["autoCorrectNormal", "axisX", "axisY", "axisZ", "axis", "axisChoice", "bridge", "bridgeCurve", "compAnchorX", "compAnchorY", "compAnchorZ", "compAnchor", "compAxisX", "compAxisY", "compAxisZ", "compAxis", "compAxisChoice", "compPivotX", "compPivotY", "compPivotZ", "compPivot", "computePivotAndAxis", "degree", "endSweep", "inputCurve", "outputSurface", "pivotX", "pivotY", "pivotZ", "pivot", "radius", "radiusAnchor", "sections", "startSweep", "tolerance", "useTolerance"]
	nodeLeafPlugs = ["autoCorrectNormal", "axis", "axisChoice", "bridge", "bridgeCurve", "compAnchor", "compAxis", "compAxisChoice", "compPivot", "computePivotAndAxis", "degree", "endSweep", "inputCurve", "outputSurface", "pivot", "radius", "radiusAnchor", "sections", "startSweep", "tolerance", "useTolerance"]
	pass

