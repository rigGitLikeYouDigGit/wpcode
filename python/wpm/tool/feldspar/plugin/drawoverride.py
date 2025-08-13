
from __future__ import annotations
"""draw override for drawing rigid structures in feldspar system"""

import sys, os

import numpy as np
from edRig.palette import *
from edRig import om, oma, omr, omui

from edRig.maya.lib import plugin
from edRig.maya.object.plugin import PluginDrawOverrideTemplate

from edRig.maya.tool.feldspar import datastruct
from edRig.maya.tool.feldspar.plugin import draw

from edRig.maya.tool.feldspar.plugin.solvernode import FeldsparSolverNode


maya_useNewAPI = True

class FeldsparDrawData(om.MUserData):

	def __init__(self, fsdata:datastruct.FeldsparData=None):
		om.MUserData.__init__(self, False)
		self.data = fsdata


class FeldsparDrawOverride(omr.MPxDrawOverride, PluginDrawOverrideTemplate):

	def __init__(self, obj):
		omr.MPxDrawOverride.__init__(self, obj, self.drawCallback, True)

	def isBounded(self, objPath, cameraPath):
		""" u cannot hold the bars """
		return False
	def supportedDrawAPIs(self):
		return omr.MRenderer.kOpenGLCoreProfile
	def hasUIDrawables(self):
		return True
	def wantUserSelection(self):
		return False


	def addUIDrawables(self, objPath, drawManager,
	                   frameContext, data:FeldsparDrawData):
		"""draw segments between joints for possible selection"""
		#solverMPx : FeldsparSolverNode = plugin.MPxNodeFromMDagPath(objPath)
		drawManager.beginDrawable(
			selectability=omr.MUIDrawManager.kNonSelectable,
			selectionName=0
		)
		draw.drawFeldsparAssembly(data.data,
		                          drawManager)


		drawManager.endDrawable()



	def prepareForDraw(self, objPath, cameraPath, frameContext, oldData):
		if not isinstance(oldData, FeldsparDrawData):
			oldData = FeldsparDrawData()
		# check if data needs rebuilding
		if not oldData.data:
			solverMPx = plugin.MPxNodeFromMDagPath(objPath)
			oldData.data = solverMPx.assembly.data
		return oldData
