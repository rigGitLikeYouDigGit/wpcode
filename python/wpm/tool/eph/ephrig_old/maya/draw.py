
from __future__ import annotations

maya_useNewAPI = True


from edRig.palette import *
from edRig import om, omui
from maya.api import OpenMayaRender as omr

#from edRig.ephrig.rig import EphRig, EphSolver
from edRig.ephrig.maya.rig import MEphRig
from edRig.ephrig.maya.node import MEphNode

from OpenGL.GL import *

if T.TYPE_CHECKING:
	from edRig.ephrig.maya.pluginnode import EphRigDrawData

def getCurrentView():
	return omui.M3dView.active3dView()

def getGLFT():
	return omr.MHardwareRenderer.theRenderer().glFunctionTable()


class GLBlock(object):
	def __init__(self, glEnum):
		self.enum = glEnum
	def __enter__(self):
		glBegin(self.enum)

	def __exit__(self, exc_type, exc_val, exc_tb):
		glEnd()
		if exc_type:
			traceback.print_exc()



def drawRigGL(frameCtx:omr.MFrameContex, drawData:"EphRigDrawData"):
	""" draw whole rig using draw data"""
	with GLBlock(GL_LINES):
		for src, dst in drawData.drawVecs:
			glVertex3f(*src)
			glVertex3f(*dst)




def drawRig(rig:MEphRig, mgr:omr.MUIDrawManager):
	"""draws a representation of single ephemeral rig """
	for i in rig.groundGraph.edges:
		drawEdge(i[0], i[1], mgr)

def drawEdge(srcNode:MEphNode, dstNode:MEphNode, mgr:omr.MUIDrawManager):
	"""draw a single edge between maya eph nodes"""
	startPos = srcNode.outTf.worldPos()
	endPos = dstNode.outTf.worldPos()
	spanVec = endPos - startPos

	midPos = om.MPoint(startPos + spanVec * 0.5)
	upVec = spanVec.normal()

	rad = 0.1
	height = spanVec.length()

	#mgr.beginDrawable()
	#mgr.line(startPos, endPos)
	mgr.capsule(om.MPoint(midPos), upVec, rad, height, 12, 12, filled=True)
	#mgr.endDrawable()

