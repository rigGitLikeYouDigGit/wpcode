

from maya import cmds
import maya.api.OpenMaya as om
from wpm.core import api

def keyBlendShapeTargets(bsNode:str):
	# get target shapes
	# set one key each frame
	api.getMFn(bsNode) #type: om.MFnDependencyNode

	shapeNames = cmds.listConnections(bsNode + ".inputTarget[0].inputTargetGroup[*].inputTargetItem[6000].inputGeomTarget", s=1, d=0)

	for i, name in enumerate(shapeNames):
		cmds.setKeyframe(bsNode + ".weight[0]", t=i, v=1)
		cmds.setKeyframe(bsNode + ".weight[0]", t=i+1, v=0)

def blendShapeTargetsFromMeshSequence(
		bsNode, meshNode,
		nameFrameMap:dict[int, str]):
	pass




