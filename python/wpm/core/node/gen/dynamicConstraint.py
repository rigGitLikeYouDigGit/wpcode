

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class BendPlug(Plug):
	node : DynamicConstraint = None
	pass
class BendBreakAnglePlug(Plug):
	node : DynamicConstraint = None
	pass
class BendStrengthPlug(Plug):
	node : DynamicConstraint = None
	pass
class CollidePlug(Plug):
	node : DynamicConstraint = None
	pass
class CollideWidthScalePlug(Plug):
	node : DynamicConstraint = None
	pass
class ComponentIdsPlug(Plug):
	node : DynamicConstraint = None
	pass
class ComponentRelationPlug(Plug):
	node : DynamicConstraint = None
	pass
class ConnectWithinComponentPlug(Plug):
	node : DynamicConstraint = None
	pass
class ConnectionDensityPlug(Plug):
	node : DynamicConstraint = None
	pass
class ConnectionDensityRange_FloatValuePlug(Plug):
	parent : ConnectionDensityRangePlug = PlugDescriptor("connectionDensityRange")
	node : DynamicConstraint = None
	pass
class ConnectionDensityRange_InterpPlug(Plug):
	parent : ConnectionDensityRangePlug = PlugDescriptor("connectionDensityRange")
	node : DynamicConstraint = None
	pass
class ConnectionDensityRange_PositionPlug(Plug):
	parent : ConnectionDensityRangePlug = PlugDescriptor("connectionDensityRange")
	node : DynamicConstraint = None
	pass
class ConnectionDensityRangePlug(Plug):
	connectionDensityRange_FloatValue_ : ConnectionDensityRange_FloatValuePlug = PlugDescriptor("connectionDensityRange_FloatValue")
	cdnrfv_ : ConnectionDensityRange_FloatValuePlug = PlugDescriptor("connectionDensityRange_FloatValue")
	connectionDensityRange_Interp_ : ConnectionDensityRange_InterpPlug = PlugDescriptor("connectionDensityRange_Interp")
	cdnri_ : ConnectionDensityRange_InterpPlug = PlugDescriptor("connectionDensityRange_Interp")
	connectionDensityRange_Position_ : ConnectionDensityRange_PositionPlug = PlugDescriptor("connectionDensityRange_Position")
	cdnrp_ : ConnectionDensityRange_PositionPlug = PlugDescriptor("connectionDensityRange_Position")
	node : DynamicConstraint = None
	pass
class ConnectionMethodPlug(Plug):
	node : DynamicConstraint = None
	pass
class ConnectionUpdatePlug(Plug):
	node : DynamicConstraint = None
	pass
class ConstraintMethodPlug(Plug):
	node : DynamicConstraint = None
	pass
class ConstraintRelationPlug(Plug):
	node : DynamicConstraint = None
	pass
class CurrentTimePlug(Plug):
	node : DynamicConstraint = None
	pass
class DampPlug(Plug):
	node : DynamicConstraint = None
	pass
class DisplayConnectionsPlug(Plug):
	node : DynamicConstraint = None
	pass
class DropoffPlug(Plug):
	node : DynamicConstraint = None
	pass
class DropoffDistancePlug(Plug):
	node : DynamicConstraint = None
	pass
class EnablePlug(Plug):
	node : DynamicConstraint = None
	pass
class EvalCurrentPlug(Plug):
	node : DynamicConstraint = None
	pass
class EvalStartPlug(Plug):
	node : DynamicConstraint = None
	pass
class ExcludeCollisionsPlug(Plug):
	node : DynamicConstraint = None
	pass
class ForcePlug(Plug):
	node : DynamicConstraint = None
	pass
class FrictionPlug(Plug):
	node : DynamicConstraint = None
	pass
class GlueStrengthPlug(Plug):
	node : DynamicConstraint = None
	pass
class GlueStrengthScalePlug(Plug):
	node : DynamicConstraint = None
	pass
class IsDynamicPlug(Plug):
	node : DynamicConstraint = None
	pass
class IterationsPlug(Plug):
	node : DynamicConstraint = None
	pass
class LocalCollidePlug(Plug):
	node : DynamicConstraint = None
	pass
class MaxDistancePlug(Plug):
	node : DynamicConstraint = None
	pass
class MaxIterationsPlug(Plug):
	node : DynamicConstraint = None
	pass
class MinIterationsPlug(Plug):
	node : DynamicConstraint = None
	pass
class MotionDragPlug(Plug):
	node : DynamicConstraint = None
	pass
class RestLengthPlug(Plug):
	node : DynamicConstraint = None
	pass
class RestLengthMethodPlug(Plug):
	node : DynamicConstraint = None
	pass
class RestLengthScalePlug(Plug):
	node : DynamicConstraint = None
	pass
class SingleSidedPlug(Plug):
	node : DynamicConstraint = None
	pass
class StrengthPlug(Plug):
	node : DynamicConstraint = None
	pass
class StrengthDropoff_FloatValuePlug(Plug):
	parent : StrengthDropoffPlug = PlugDescriptor("strengthDropoff")
	node : DynamicConstraint = None
	pass
class StrengthDropoff_InterpPlug(Plug):
	parent : StrengthDropoffPlug = PlugDescriptor("strengthDropoff")
	node : DynamicConstraint = None
	pass
class StrengthDropoff_PositionPlug(Plug):
	parent : StrengthDropoffPlug = PlugDescriptor("strengthDropoff")
	node : DynamicConstraint = None
	pass
class StrengthDropoffPlug(Plug):
	strengthDropoff_FloatValue_ : StrengthDropoff_FloatValuePlug = PlugDescriptor("strengthDropoff_FloatValue")
	sdpfv_ : StrengthDropoff_FloatValuePlug = PlugDescriptor("strengthDropoff_FloatValue")
	strengthDropoff_Interp_ : StrengthDropoff_InterpPlug = PlugDescriptor("strengthDropoff_Interp")
	sdpi_ : StrengthDropoff_InterpPlug = PlugDescriptor("strengthDropoff_Interp")
	strengthDropoff_Position_ : StrengthDropoff_PositionPlug = PlugDescriptor("strengthDropoff_Position")
	sdpp_ : StrengthDropoff_PositionPlug = PlugDescriptor("strengthDropoff_Position")
	node : DynamicConstraint = None
	pass
class TangentStrengthPlug(Plug):
	node : DynamicConstraint = None
	pass
# endregion


# define node class
class DynamicConstraint(Shape):
	bend_ : BendPlug = PlugDescriptor("bend")
	bendBreakAngle_ : BendBreakAnglePlug = PlugDescriptor("bendBreakAngle")
	bendStrength_ : BendStrengthPlug = PlugDescriptor("bendStrength")
	collide_ : CollidePlug = PlugDescriptor("collide")
	collideWidthScale_ : CollideWidthScalePlug = PlugDescriptor("collideWidthScale")
	componentIds_ : ComponentIdsPlug = PlugDescriptor("componentIds")
	componentRelation_ : ComponentRelationPlug = PlugDescriptor("componentRelation")
	connectWithinComponent_ : ConnectWithinComponentPlug = PlugDescriptor("connectWithinComponent")
	connectionDensity_ : ConnectionDensityPlug = PlugDescriptor("connectionDensity")
	connectionDensityRange_FloatValue_ : ConnectionDensityRange_FloatValuePlug = PlugDescriptor("connectionDensityRange_FloatValue")
	connectionDensityRange_Interp_ : ConnectionDensityRange_InterpPlug = PlugDescriptor("connectionDensityRange_Interp")
	connectionDensityRange_Position_ : ConnectionDensityRange_PositionPlug = PlugDescriptor("connectionDensityRange_Position")
	connectionDensityRange_ : ConnectionDensityRangePlug = PlugDescriptor("connectionDensityRange")
	connectionMethod_ : ConnectionMethodPlug = PlugDescriptor("connectionMethod")
	connectionUpdate_ : ConnectionUpdatePlug = PlugDescriptor("connectionUpdate")
	constraintMethod_ : ConstraintMethodPlug = PlugDescriptor("constraintMethod")
	constraintRelation_ : ConstraintRelationPlug = PlugDescriptor("constraintRelation")
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	damp_ : DampPlug = PlugDescriptor("damp")
	displayConnections_ : DisplayConnectionsPlug = PlugDescriptor("displayConnections")
	dropoff_ : DropoffPlug = PlugDescriptor("dropoff")
	dropoffDistance_ : DropoffDistancePlug = PlugDescriptor("dropoffDistance")
	enable_ : EnablePlug = PlugDescriptor("enable")
	evalCurrent_ : EvalCurrentPlug = PlugDescriptor("evalCurrent")
	evalStart_ : EvalStartPlug = PlugDescriptor("evalStart")
	excludeCollisions_ : ExcludeCollisionsPlug = PlugDescriptor("excludeCollisions")
	force_ : ForcePlug = PlugDescriptor("force")
	friction_ : FrictionPlug = PlugDescriptor("friction")
	glueStrength_ : GlueStrengthPlug = PlugDescriptor("glueStrength")
	glueStrengthScale_ : GlueStrengthScalePlug = PlugDescriptor("glueStrengthScale")
	isDynamic_ : IsDynamicPlug = PlugDescriptor("isDynamic")
	iterations_ : IterationsPlug = PlugDescriptor("iterations")
	localCollide_ : LocalCollidePlug = PlugDescriptor("localCollide")
	maxDistance_ : MaxDistancePlug = PlugDescriptor("maxDistance")
	maxIterations_ : MaxIterationsPlug = PlugDescriptor("maxIterations")
	minIterations_ : MinIterationsPlug = PlugDescriptor("minIterations")
	motionDrag_ : MotionDragPlug = PlugDescriptor("motionDrag")
	restLength_ : RestLengthPlug = PlugDescriptor("restLength")
	restLengthMethod_ : RestLengthMethodPlug = PlugDescriptor("restLengthMethod")
	restLengthScale_ : RestLengthScalePlug = PlugDescriptor("restLengthScale")
	singleSided_ : SingleSidedPlug = PlugDescriptor("singleSided")
	strength_ : StrengthPlug = PlugDescriptor("strength")
	strengthDropoff_FloatValue_ : StrengthDropoff_FloatValuePlug = PlugDescriptor("strengthDropoff_FloatValue")
	strengthDropoff_Interp_ : StrengthDropoff_InterpPlug = PlugDescriptor("strengthDropoff_Interp")
	strengthDropoff_Position_ : StrengthDropoff_PositionPlug = PlugDescriptor("strengthDropoff_Position")
	strengthDropoff_ : StrengthDropoffPlug = PlugDescriptor("strengthDropoff")
	tangentStrength_ : TangentStrengthPlug = PlugDescriptor("tangentStrength")

	# node attributes

	typeName = "dynamicConstraint"
	apiTypeInt = 993
	apiTypeStr = "kDynamicConstraint"
	typeIdInt = 1145261902
	MFnCls = om.MFnDagNode
	pass

