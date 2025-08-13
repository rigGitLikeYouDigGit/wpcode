

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
class CurvaturePlug(Plug):
	node : ProjectTangent = None
	pass
class CurvatureScalePlug(Plug):
	node : ProjectTangent = None
	pass
class IgnoreEdgesPlug(Plug):
	node : ProjectTangent = None
	pass
class InputCurve1ToProjectToPlug(Plug):
	node : ProjectTangent = None
	pass
class InputCurve2ToProjectToPlug(Plug):
	node : ProjectTangent = None
	pass
class InputCurveToProjectPlug(Plug):
	node : ProjectTangent = None
	pass
class InputSurfaceToProjectToPlug(Plug):
	node : ProjectTangent = None
	pass
class OutputCurvePlug(Plug):
	node : ProjectTangent = None
	pass
class ReverseTangentPlug(Plug):
	node : ProjectTangent = None
	pass
class RotatePlug(Plug):
	node : ProjectTangent = None
	pass
class TangentDirectionPlug(Plug):
	node : ProjectTangent = None
	pass
class TangentScalePlug(Plug):
	node : ProjectTangent = None
	pass
# endregion


# define node class
class ProjectTangent(AbstractBaseCreate):
	curvature_ : CurvaturePlug = PlugDescriptor("curvature")
	curvatureScale_ : CurvatureScalePlug = PlugDescriptor("curvatureScale")
	ignoreEdges_ : IgnoreEdgesPlug = PlugDescriptor("ignoreEdges")
	inputCurve1ToProjectTo_ : InputCurve1ToProjectToPlug = PlugDescriptor("inputCurve1ToProjectTo")
	inputCurve2ToProjectTo_ : InputCurve2ToProjectToPlug = PlugDescriptor("inputCurve2ToProjectTo")
	inputCurveToProject_ : InputCurveToProjectPlug = PlugDescriptor("inputCurveToProject")
	inputSurfaceToProjectTo_ : InputSurfaceToProjectToPlug = PlugDescriptor("inputSurfaceToProjectTo")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	reverseTangent_ : ReverseTangentPlug = PlugDescriptor("reverseTangent")
	rotate_ : RotatePlug = PlugDescriptor("rotate")
	tangentDirection_ : TangentDirectionPlug = PlugDescriptor("tangentDirection")
	tangentScale_ : TangentScalePlug = PlugDescriptor("tangentScale")

	# node attributes

	typeName = "projectTangent"
	apiTypeInt = 88
	apiTypeStr = "kProjectTangent"
	typeIdInt = 1313887310
	MFnCls = om.MFnDependencyNode
	pass

