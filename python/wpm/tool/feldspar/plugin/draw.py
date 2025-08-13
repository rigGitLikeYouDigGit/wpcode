
from __future__ import annotations
"""functions or drawing a feldspar network"""
from edRig.palette import *
from edRig import cmds, om, omr, ECA, EdNode

from edRig.maya.tool.feldspar import assembly, datastruct

def drawFeldsparAssembly(data:datastruct.FeldsparData, mgr:omr.MUIDrawManager):
	"""Draw full capsules for rigid rods, dashed lines for soft ones"""
	for barData in data.barDatas:
		startPoint = data.positions[barData.indices[0]]
		endPoint = data.positions[barData.indices[1]]
		spanVec = endPoint - startPoint
		midPos = om.MPoint(startPoint + spanVec * 0.5)
		upVec = om.MVector(spanVec).normal()

		rad = 0.1
		subdivHeight = 4
		subdivAxis = 4
		filled = not barData.soft

		mgr.capsule(om.MPoint(midPos), upVec, rad,
		            om.MVector(spanVec).length(),
		            subdivAxis, subdivHeight, filled)



# def drawEdge(srcNode:MEphNode, dstNode:MEphNode, mgr:omr.MUIDrawManager):
# 	"""draw a single edge between maya eph nodes"""
# 	startPos = srcNode.outTf.worldPos()
# 	endPos = dstNode.outTf.worldPos()
# 	spanVec = endPos - startPos
#
# 	midPos = om.MPoint(startPos + spanVec * 0.5)
# 	upVec = spanVec.normal()
#
# 	rad = 0.1
# 	height = spanVec.length()
#
# 	#mgr.beginDrawable()
# 	#mgr.line(startPos, endPos)
# 	mgr.capsule(om.MPoint(midPos), upVec, rad, height, 12, 12, filled=True)
# 	#mgr.endDrawable()
