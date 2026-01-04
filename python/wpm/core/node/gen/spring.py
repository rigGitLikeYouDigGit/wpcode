

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Shape = Catalogue.Shape
else:
	from .. import retriever
	Shape = retriever.getNodeCls("Shape")
	assert Shape

# add node doc



# region plug type defs
class CountPlug(Plug):
	node : Spring = None
	pass
class DampingPlug(Plug):
	node : Spring = None
	pass
class DampingPSPlug(Plug):
	node : Spring = None
	pass
class DeltaTimePlug(Plug):
	node : Spring = None
	pass
class End1WeightPlug(Plug):
	node : Spring = None
	pass
class End2WeightPlug(Plug):
	node : Spring = None
	pass
class IdIndexPlug(Plug):
	parent : IdMappingPlug = PlugDescriptor("idMapping")
	node : Spring = None
	pass
class SortedIdPlug(Plug):
	parent : IdMappingPlug = PlugDescriptor("idMapping")
	node : Spring = None
	pass
class IdMappingPlug(Plug):
	idIndex_ : IdIndexPlug = PlugDescriptor("idIndex")
	idix_ : IdIndexPlug = PlugDescriptor("idIndex")
	sortedId_ : SortedIdPlug = PlugDescriptor("sortedId")
	sid_ : SortedIdPlug = PlugDescriptor("sortedId")
	node : Spring = None
	pass
class LengthsPlug(Plug):
	node : Spring = None
	pass
class ManageParticleDeathPlug(Plug):
	node : Spring = None
	pass
class MaxUsedPlug(Plug):
	node : Spring = None
	pass
class MinSpringsPlug(Plug):
	node : Spring = None
	pass
class MinUsedPlug(Plug):
	node : Spring = None
	pass
class Obj0IndexPlug(Plug):
	node : Spring = None
	pass
class Obj1IndexPlug(Plug):
	node : Spring = None
	pass
class ObjCountPlug(Plug):
	node : Spring = None
	pass
class Object0Plug(Plug):
	node : Spring = None
	pass
class Object1Plug(Plug):
	node : Spring = None
	pass
class ObjectMassPlug(Plug):
	node : Spring = None
	pass
class ObjectPositionsPlug(Plug):
	node : Spring = None
	pass
class ObjectVelocitiesPlug(Plug):
	node : Spring = None
	pass
class ObjectsPlug(Plug):
	node : Spring = None
	pass
class OutputForcePlug(Plug):
	node : Spring = None
	pass
class Point0Plug(Plug):
	node : Spring = None
	pass
class Point1Plug(Plug):
	node : Spring = None
	pass
class Pt0IndexPlug(Plug):
	node : Spring = None
	pass
class Pt1IndexPlug(Plug):
	node : Spring = None
	pass
class RestLengthPlug(Plug):
	node : Spring = None
	pass
class RestLengthPSPlug(Plug):
	node : Spring = None
	pass
class StiffnessPlug(Plug):
	node : Spring = None
	pass
class StiffnessPSPlug(Plug):
	node : Spring = None
	pass
class UseDampingPSPlug(Plug):
	node : Spring = None
	pass
class UseRestLengthPSPlug(Plug):
	node : Spring = None
	pass
class UseStiffnessPSPlug(Plug):
	node : Spring = None
	pass
class ValidIndexPlug(Plug):
	node : Spring = None
	pass
# endregion


# define node class
class Spring(Shape):
	count_ : CountPlug = PlugDescriptor("count")
	damping_ : DampingPlug = PlugDescriptor("damping")
	dampingPS_ : DampingPSPlug = PlugDescriptor("dampingPS")
	deltaTime_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	end1Weight_ : End1WeightPlug = PlugDescriptor("end1Weight")
	end2Weight_ : End2WeightPlug = PlugDescriptor("end2Weight")
	idIndex_ : IdIndexPlug = PlugDescriptor("idIndex")
	sortedId_ : SortedIdPlug = PlugDescriptor("sortedId")
	idMapping_ : IdMappingPlug = PlugDescriptor("idMapping")
	lengths_ : LengthsPlug = PlugDescriptor("lengths")
	manageParticleDeath_ : ManageParticleDeathPlug = PlugDescriptor("manageParticleDeath")
	maxUsed_ : MaxUsedPlug = PlugDescriptor("maxUsed")
	minSprings_ : MinSpringsPlug = PlugDescriptor("minSprings")
	minUsed_ : MinUsedPlug = PlugDescriptor("minUsed")
	obj0Index_ : Obj0IndexPlug = PlugDescriptor("obj0Index")
	obj1Index_ : Obj1IndexPlug = PlugDescriptor("obj1Index")
	objCount_ : ObjCountPlug = PlugDescriptor("objCount")
	object0_ : Object0Plug = PlugDescriptor("object0")
	object1_ : Object1Plug = PlugDescriptor("object1")
	objectMass_ : ObjectMassPlug = PlugDescriptor("objectMass")
	objectPositions_ : ObjectPositionsPlug = PlugDescriptor("objectPositions")
	objectVelocities_ : ObjectVelocitiesPlug = PlugDescriptor("objectVelocities")
	objects_ : ObjectsPlug = PlugDescriptor("objects")
	outputForce_ : OutputForcePlug = PlugDescriptor("outputForce")
	point0_ : Point0Plug = PlugDescriptor("point0")
	point1_ : Point1Plug = PlugDescriptor("point1")
	pt0Index_ : Pt0IndexPlug = PlugDescriptor("pt0Index")
	pt1Index_ : Pt1IndexPlug = PlugDescriptor("pt1Index")
	restLength_ : RestLengthPlug = PlugDescriptor("restLength")
	restLengthPS_ : RestLengthPSPlug = PlugDescriptor("restLengthPS")
	stiffness_ : StiffnessPlug = PlugDescriptor("stiffness")
	stiffnessPS_ : StiffnessPSPlug = PlugDescriptor("stiffnessPS")
	useDampingPS_ : UseDampingPSPlug = PlugDescriptor("useDampingPS")
	useRestLengthPS_ : UseRestLengthPSPlug = PlugDescriptor("useRestLengthPS")
	useStiffnessPS_ : UseStiffnessPSPlug = PlugDescriptor("useStiffnessPS")
	validIndex_ : ValidIndexPlug = PlugDescriptor("validIndex")

	# node attributes

	typeName = "spring"
	apiTypeInt = 315
	apiTypeStr = "kSpring"
	typeIdInt = 1498632274
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["count", "damping", "dampingPS", "deltaTime", "end1Weight", "end2Weight", "idIndex", "sortedId", "idMapping", "lengths", "manageParticleDeath", "maxUsed", "minSprings", "minUsed", "obj0Index", "obj1Index", "objCount", "object0", "object1", "objectMass", "objectPositions", "objectVelocities", "objects", "outputForce", "point0", "point1", "pt0Index", "pt1Index", "restLength", "restLengthPS", "stiffness", "stiffnessPS", "useDampingPS", "useRestLengthPS", "useStiffnessPS", "validIndex"]
	nodeLeafPlugs = ["count", "damping", "dampingPS", "deltaTime", "end1Weight", "end2Weight", "idMapping", "lengths", "manageParticleDeath", "maxUsed", "minSprings", "minUsed", "obj0Index", "obj1Index", "objCount", "object0", "object1", "objectMass", "objectPositions", "objectVelocities", "objects", "outputForce", "point0", "point1", "pt0Index", "pt1Index", "restLength", "restLengthPS", "stiffness", "stiffnessPS", "useDampingPS", "useRestLengthPS", "useStiffnessPS", "validIndex"]
	pass

