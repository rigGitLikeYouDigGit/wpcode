

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Cluster = retriever.getNodeCls("Cluster")
assert Cluster
if T.TYPE_CHECKING:
	from .. import Cluster

# add node doc



# region plug type defs
class BindPosePlug(Plug):
	node : JointCluster = None
	pass
class BoneLengthPlug(Plug):
	node : JointCluster = None
	pass
class ChildEnabledPlug(Plug):
	node : JointCluster = None
	pass
class ChildJointBindPosePlug(Plug):
	node : JointCluster = None
	pass
class ChildJointPostMatrixPlug(Plug):
	parent : ChildJointClusterXformsPlug = PlugDescriptor("childJointClusterXforms")
	node : JointCluster = None
	pass
class ChildJointPreMatrixPlug(Plug):
	parent : ChildJointClusterXformsPlug = PlugDescriptor("childJointClusterXforms")
	node : JointCluster = None
	pass
class ChildJointWeightedMatrixPlug(Plug):
	parent : ChildJointClusterXformsPlug = PlugDescriptor("childJointClusterXforms")
	node : JointCluster = None
	pass
class ChildJointClusterXformsPlug(Plug):
	childJointPostMatrix_ : ChildJointPostMatrixPlug = PlugDescriptor("childJointPostMatrix")
	cpo_ : ChildJointPostMatrixPlug = PlugDescriptor("childJointPostMatrix")
	childJointPreMatrix_ : ChildJointPreMatrixPlug = PlugDescriptor("childJointPreMatrix")
	cpr_ : ChildJointPreMatrixPlug = PlugDescriptor("childJointPreMatrix")
	childJointWeightedMatrix_ : ChildJointWeightedMatrixPlug = PlugDescriptor("childJointWeightedMatrix")
	cjw_ : ChildJointWeightedMatrixPlug = PlugDescriptor("childJointWeightedMatrix")
	node : JointCluster = None
	pass
class ChildjointMidplaneAxisXPlug(Plug):
	parent : ChildJointMidplaneAxisPlug = PlugDescriptor("childJointMidplaneAxis")
	node : JointCluster = None
	pass
class ChildjointMidplaneAxisYPlug(Plug):
	parent : ChildJointMidplaneAxisPlug = PlugDescriptor("childJointMidplaneAxis")
	node : JointCluster = None
	pass
class ChildjointMidplaneAxisZPlug(Plug):
	parent : ChildJointMidplaneAxisPlug = PlugDescriptor("childJointMidplaneAxis")
	node : JointCluster = None
	pass
class ChildJointMidplaneAxisPlug(Plug):
	childjointMidplaneAxisX_ : ChildjointMidplaneAxisXPlug = PlugDescriptor("childjointMidplaneAxisX")
	cmx_ : ChildjointMidplaneAxisXPlug = PlugDescriptor("childjointMidplaneAxisX")
	childjointMidplaneAxisY_ : ChildjointMidplaneAxisYPlug = PlugDescriptor("childjointMidplaneAxisY")
	cmy_ : ChildjointMidplaneAxisYPlug = PlugDescriptor("childjointMidplaneAxisY")
	childjointMidplaneAxisZ_ : ChildjointMidplaneAxisZPlug = PlugDescriptor("childjointMidplaneAxisZ")
	cmz_ : ChildjointMidplaneAxisZPlug = PlugDescriptor("childjointMidplaneAxisZ")
	node : JointCluster = None
	pass
class ChildJointPostCompensationMatrixPlug(Plug):
	node : JointCluster = None
	pass
class ChildJointPreCompensationMatrixPlug(Plug):
	node : JointCluster = None
	pass
class ChildJointWeightedCompensationMatrixPlug(Plug):
	node : JointCluster = None
	pass
class ClusterFlexorSetPlug(Plug):
	node : JointCluster = None
	pass
class ConvertedTo2Plug(Plug):
	node : JointCluster = None
	pass
class DistancesPlug(Plug):
	parent : DistanceListPlug = PlugDescriptor("distanceList")
	node : JointCluster = None
	pass
class DistanceListPlug(Plug):
	distances_ : DistancesPlug = PlugDescriptor("distances")
	cd_ : DistancesPlug = PlugDescriptor("distances")
	node : JointCluster = None
	pass
class EnableAutoPercentUpdatePlug(Plug):
	node : JointCluster = None
	pass
class JointMidplaneAxisXPlug(Plug):
	parent : JointMidplaneAxisPlug = PlugDescriptor("jointMidplaneAxis")
	node : JointCluster = None
	pass
class JointMidplaneAxisYPlug(Plug):
	parent : JointMidplaneAxisPlug = PlugDescriptor("jointMidplaneAxis")
	node : JointCluster = None
	pass
class JointMidplaneAxisZPlug(Plug):
	parent : JointMidplaneAxisPlug = PlugDescriptor("jointMidplaneAxis")
	node : JointCluster = None
	pass
class JointMidplaneAxisPlug(Plug):
	jointMidplaneAxisX_ : JointMidplaneAxisXPlug = PlugDescriptor("jointMidplaneAxisX")
	jmx_ : JointMidplaneAxisXPlug = PlugDescriptor("jointMidplaneAxisX")
	jointMidplaneAxisY_ : JointMidplaneAxisYPlug = PlugDescriptor("jointMidplaneAxisY")
	jmy_ : JointMidplaneAxisYPlug = PlugDescriptor("jointMidplaneAxisY")
	jointMidplaneAxisZ_ : JointMidplaneAxisZPlug = PlugDescriptor("jointMidplaneAxisZ")
	jmz_ : JointMidplaneAxisZPlug = PlugDescriptor("jointMidplaneAxisZ")
	node : JointCluster = None
	pass
class LastLowerBoundPlug(Plug):
	node : JointCluster = None
	pass
class LastUpperBoundPlug(Plug):
	node : JointCluster = None
	pass
class LowerBoundPlug(Plug):
	node : JointCluster = None
	pass
class LowerDropoffTypePlug(Plug):
	node : JointCluster = None
	pass
class LowerEnabledPlug(Plug):
	node : JointCluster = None
	pass
class LowerValuePlug(Plug):
	node : JointCluster = None
	pass
class NextJointBindPosePlug(Plug):
	node : JointCluster = None
	pass
class NextJointPostMatrixPlug(Plug):
	parent : NextJointClusterXformsPlug = PlugDescriptor("nextJointClusterXforms")
	node : JointCluster = None
	pass
class NextJointPreMatrixPlug(Plug):
	parent : NextJointClusterXformsPlug = PlugDescriptor("nextJointClusterXforms")
	node : JointCluster = None
	pass
class NextJointWeightedMatrixPlug(Plug):
	parent : NextJointClusterXformsPlug = PlugDescriptor("nextJointClusterXforms")
	node : JointCluster = None
	pass
class NextJointClusterXformsPlug(Plug):
	nextJointPostMatrix_ : NextJointPostMatrixPlug = PlugDescriptor("nextJointPostMatrix")
	npo_ : NextJointPostMatrixPlug = PlugDescriptor("nextJointPostMatrix")
	nextJointPreMatrix_ : NextJointPreMatrixPlug = PlugDescriptor("nextJointPreMatrix")
	npr_ : NextJointPreMatrixPlug = PlugDescriptor("nextJointPreMatrix")
	nextJointWeightedMatrix_ : NextJointWeightedMatrixPlug = PlugDescriptor("nextJointWeightedMatrix")
	njw_ : NextJointWeightedMatrixPlug = PlugDescriptor("nextJointWeightedMatrix")
	node : JointCluster = None
	pass
class NextjointMidplaneAxisXPlug(Plug):
	parent : NextJointMidplaneAxisPlug = PlugDescriptor("nextJointMidplaneAxis")
	node : JointCluster = None
	pass
class NextjointMidplaneAxisYPlug(Plug):
	parent : NextJointMidplaneAxisPlug = PlugDescriptor("nextJointMidplaneAxis")
	node : JointCluster = None
	pass
class NextjointMidplaneAxisZPlug(Plug):
	parent : NextJointMidplaneAxisPlug = PlugDescriptor("nextJointMidplaneAxis")
	node : JointCluster = None
	pass
class NextJointMidplaneAxisPlug(Plug):
	nextjointMidplaneAxisX_ : NextjointMidplaneAxisXPlug = PlugDescriptor("nextjointMidplaneAxisX")
	nmx_ : NextjointMidplaneAxisXPlug = PlugDescriptor("nextjointMidplaneAxisX")
	nextjointMidplaneAxisY_ : NextjointMidplaneAxisYPlug = PlugDescriptor("nextjointMidplaneAxisY")
	nmy_ : NextjointMidplaneAxisYPlug = PlugDescriptor("nextjointMidplaneAxisY")
	nextjointMidplaneAxisZ_ : NextjointMidplaneAxisZPlug = PlugDescriptor("nextjointMidplaneAxisZ")
	nmz_ : NextjointMidplaneAxisZPlug = PlugDescriptor("nextjointMidplaneAxisZ")
	node : JointCluster = None
	pass
class NextJointPostCompensationMatrixPlug(Plug):
	node : JointCluster = None
	pass
class NextJointPreCompensationMatrixPlug(Plug):
	node : JointCluster = None
	pass
class NextJointWeightedCompensationMatrixPlug(Plug):
	node : JointCluster = None
	pass
class RedoLowerWeightsPlug(Plug):
	node : JointCluster = None
	pass
class RedoUpperWeightsPlug(Plug):
	node : JointCluster = None
	pass
class UpperBoundPlug(Plug):
	node : JointCluster = None
	pass
class UpperDropoffTypePlug(Plug):
	node : JointCluster = None
	pass
class UpperEnabledPlug(Plug):
	node : JointCluster = None
	pass
class UpperValuePlug(Plug):
	node : JointCluster = None
	pass
# endregion


# define node class
class JointCluster(Cluster):
	bindPose_ : BindPosePlug = PlugDescriptor("bindPose")
	boneLength_ : BoneLengthPlug = PlugDescriptor("boneLength")
	childEnabled_ : ChildEnabledPlug = PlugDescriptor("childEnabled")
	childJointBindPose_ : ChildJointBindPosePlug = PlugDescriptor("childJointBindPose")
	childJointPostMatrix_ : ChildJointPostMatrixPlug = PlugDescriptor("childJointPostMatrix")
	childJointPreMatrix_ : ChildJointPreMatrixPlug = PlugDescriptor("childJointPreMatrix")
	childJointWeightedMatrix_ : ChildJointWeightedMatrixPlug = PlugDescriptor("childJointWeightedMatrix")
	childJointClusterXforms_ : ChildJointClusterXformsPlug = PlugDescriptor("childJointClusterXforms")
	childjointMidplaneAxisX_ : ChildjointMidplaneAxisXPlug = PlugDescriptor("childjointMidplaneAxisX")
	childjointMidplaneAxisY_ : ChildjointMidplaneAxisYPlug = PlugDescriptor("childjointMidplaneAxisY")
	childjointMidplaneAxisZ_ : ChildjointMidplaneAxisZPlug = PlugDescriptor("childjointMidplaneAxisZ")
	childJointMidplaneAxis_ : ChildJointMidplaneAxisPlug = PlugDescriptor("childJointMidplaneAxis")
	childJointPostCompensationMatrix_ : ChildJointPostCompensationMatrixPlug = PlugDescriptor("childJointPostCompensationMatrix")
	childJointPreCompensationMatrix_ : ChildJointPreCompensationMatrixPlug = PlugDescriptor("childJointPreCompensationMatrix")
	childJointWeightedCompensationMatrix_ : ChildJointWeightedCompensationMatrixPlug = PlugDescriptor("childJointWeightedCompensationMatrix")
	clusterFlexorSet_ : ClusterFlexorSetPlug = PlugDescriptor("clusterFlexorSet")
	convertedTo2_ : ConvertedTo2Plug = PlugDescriptor("convertedTo2")
	distances_ : DistancesPlug = PlugDescriptor("distances")
	distanceList_ : DistanceListPlug = PlugDescriptor("distanceList")
	enableAutoPercentUpdate_ : EnableAutoPercentUpdatePlug = PlugDescriptor("enableAutoPercentUpdate")
	jointMidplaneAxisX_ : JointMidplaneAxisXPlug = PlugDescriptor("jointMidplaneAxisX")
	jointMidplaneAxisY_ : JointMidplaneAxisYPlug = PlugDescriptor("jointMidplaneAxisY")
	jointMidplaneAxisZ_ : JointMidplaneAxisZPlug = PlugDescriptor("jointMidplaneAxisZ")
	jointMidplaneAxis_ : JointMidplaneAxisPlug = PlugDescriptor("jointMidplaneAxis")
	lastLowerBound_ : LastLowerBoundPlug = PlugDescriptor("lastLowerBound")
	lastUpperBound_ : LastUpperBoundPlug = PlugDescriptor("lastUpperBound")
	lowerBound_ : LowerBoundPlug = PlugDescriptor("lowerBound")
	lowerDropoffType_ : LowerDropoffTypePlug = PlugDescriptor("lowerDropoffType")
	lowerEnabled_ : LowerEnabledPlug = PlugDescriptor("lowerEnabled")
	lowerValue_ : LowerValuePlug = PlugDescriptor("lowerValue")
	nextJointBindPose_ : NextJointBindPosePlug = PlugDescriptor("nextJointBindPose")
	nextJointPostMatrix_ : NextJointPostMatrixPlug = PlugDescriptor("nextJointPostMatrix")
	nextJointPreMatrix_ : NextJointPreMatrixPlug = PlugDescriptor("nextJointPreMatrix")
	nextJointWeightedMatrix_ : NextJointWeightedMatrixPlug = PlugDescriptor("nextJointWeightedMatrix")
	nextJointClusterXforms_ : NextJointClusterXformsPlug = PlugDescriptor("nextJointClusterXforms")
	nextjointMidplaneAxisX_ : NextjointMidplaneAxisXPlug = PlugDescriptor("nextjointMidplaneAxisX")
	nextjointMidplaneAxisY_ : NextjointMidplaneAxisYPlug = PlugDescriptor("nextjointMidplaneAxisY")
	nextjointMidplaneAxisZ_ : NextjointMidplaneAxisZPlug = PlugDescriptor("nextjointMidplaneAxisZ")
	nextJointMidplaneAxis_ : NextJointMidplaneAxisPlug = PlugDescriptor("nextJointMidplaneAxis")
	nextJointPostCompensationMatrix_ : NextJointPostCompensationMatrixPlug = PlugDescriptor("nextJointPostCompensationMatrix")
	nextJointPreCompensationMatrix_ : NextJointPreCompensationMatrixPlug = PlugDescriptor("nextJointPreCompensationMatrix")
	nextJointWeightedCompensationMatrix_ : NextJointWeightedCompensationMatrixPlug = PlugDescriptor("nextJointWeightedCompensationMatrix")
	redoLowerWeights_ : RedoLowerWeightsPlug = PlugDescriptor("redoLowerWeights")
	redoUpperWeights_ : RedoUpperWeightsPlug = PlugDescriptor("redoUpperWeights")
	upperBound_ : UpperBoundPlug = PlugDescriptor("upperBound")
	upperDropoffType_ : UpperDropoffTypePlug = PlugDescriptor("upperDropoffType")
	upperEnabled_ : UpperEnabledPlug = PlugDescriptor("upperEnabled")
	upperValue_ : UpperValuePlug = PlugDescriptor("upperValue")

	# node attributes

	typeName = "jointCluster"
	apiTypeInt = 349
	apiTypeStr = "kJointCluster"
	typeIdInt = 1179272012
	MFnCls = om.MFnGeometryFilter
	nodeLeafClassAttrs = ["bindPose", "boneLength", "childEnabled", "childJointBindPose", "childJointPostMatrix", "childJointPreMatrix", "childJointWeightedMatrix", "childJointClusterXforms", "childjointMidplaneAxisX", "childjointMidplaneAxisY", "childjointMidplaneAxisZ", "childJointMidplaneAxis", "childJointPostCompensationMatrix", "childJointPreCompensationMatrix", "childJointWeightedCompensationMatrix", "clusterFlexorSet", "convertedTo2", "distances", "distanceList", "enableAutoPercentUpdate", "jointMidplaneAxisX", "jointMidplaneAxisY", "jointMidplaneAxisZ", "jointMidplaneAxis", "lastLowerBound", "lastUpperBound", "lowerBound", "lowerDropoffType", "lowerEnabled", "lowerValue", "nextJointBindPose", "nextJointPostMatrix", "nextJointPreMatrix", "nextJointWeightedMatrix", "nextJointClusterXforms", "nextjointMidplaneAxisX", "nextjointMidplaneAxisY", "nextjointMidplaneAxisZ", "nextJointMidplaneAxis", "nextJointPostCompensationMatrix", "nextJointPreCompensationMatrix", "nextJointWeightedCompensationMatrix", "redoLowerWeights", "redoUpperWeights", "upperBound", "upperDropoffType", "upperEnabled", "upperValue"]
	nodeLeafPlugs = ["bindPose", "boneLength", "childEnabled", "childJointBindPose", "childJointClusterXforms", "childJointMidplaneAxis", "childJointPostCompensationMatrix", "childJointPreCompensationMatrix", "childJointWeightedCompensationMatrix", "clusterFlexorSet", "convertedTo2", "distanceList", "enableAutoPercentUpdate", "jointMidplaneAxis", "lastLowerBound", "lastUpperBound", "lowerBound", "lowerDropoffType", "lowerEnabled", "lowerValue", "nextJointBindPose", "nextJointClusterXforms", "nextJointMidplaneAxis", "nextJointPostCompensationMatrix", "nextJointPreCompensationMatrix", "nextJointWeightedCompensationMatrix", "redoLowerWeights", "redoUpperWeights", "upperBound", "upperDropoffType", "upperEnabled", "upperValue"]
	pass

