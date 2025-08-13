

"""draw override for drawing an EphRigNode's presence in scene"""

import sys, os

from itertools import product

from maya import cmds
from maya.api import OpenMaya as om, OpenMayaUI as omui, OpenMayaRender as omr

from edRig.ephrig.maya.node import MEphNode
from edRig.ephrig.rig import EphRig
from edRig.ephrig.maya import util, lib, draw

from edRig.ephrig.maya.pluginnode import EphRigDrawData

maya_useNewAPI = True



def ephDrawCallback(*args, **kwargs):
	"""args are MFrameContext, EphRigDrawData"""
	frameCtx, drawData = args
	if drawData is None:
		print("draw data none")
		return
	draw.drawRigGL(frameCtx, drawData)


class EphRigDrawOverride(omr.MPxDrawOverride):

	def __init__(self, obj):
		omr.MPxDrawOverride.__init__(self, obj, ephDrawCallback, True)

	@classmethod
	def creator(cls, obj):
		""" create new rig object and assign it to the node"""
		newNode = cls(obj)
		return newNode

	@staticmethod
	def pyObjFromObjectPath(objPath):
		nodeFn = om.MFnDagNode(objPath)
		pyObj = lib.ephPyObject(nodeFn.object())
		return pyObj

	def isBounded(self, objPath, cameraPath):
		""" u cannot hold the rig """
		return False
	def supportedDrawAPIs(self):
		return omr.MRenderer.kOpenGLCoreProfile
	def hasUIDrawables(self):
		return True
	def wantUserSelection(self):
		return True

	def userSelect(self,
	               selectInfo:omr.MSelectionInfo,
	               drawCtx:omr.MDrawContext,
	               objPath:om.MDagPath,
	               data:EphRigDrawData,
	               selectionList:om.MSelectionList,
	               worldSpaceHitPts:om.MPointArray
	               ):
		return False

	def addUIDrawables(self, objPath, drawManager,
	                   frameContext, data:EphRigDrawData):
		"""draw segments between joints for possible selection"""
		pyObj = self.pyObjFromObjectPath(objPath)
		drawManager.beginDrawable(
			#selectability=omr.MUIDrawManager.kAutomatic,
			selectability=omr.MUIDrawManager.kNonSelectable,
			selectionName=0
		)
		rig = pyObj.ephRig
		for i in rig.groundGraph.edges:
			draw.drawEdge(i[0], i[1], drawManager)

		drawManager.endDrawable()



	def prepareForDraw(self, objPath, cameraPath, frameContext, oldData):
		if not isinstance(oldData, EphRigDrawData):
			oldData = EphRigDrawData()
		if not oldData.rig:
			pyObj = self.pyObjFromObjectPath(objPath)
			oldData.rig = pyObj.ephRig
		oldData.drawVecs = oldData.rig.edgeVectors()

		return oldData
