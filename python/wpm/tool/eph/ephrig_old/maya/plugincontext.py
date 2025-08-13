maya_useNewAPI = True

import sys, os

from itertools import product

from maya import cmds
from maya.api import OpenMaya as om, OpenMayaUI as omui

from edRig.ephrig.maya.node import MEphNode
from edRig.ephrig.rig import EphRig
from edRig.ephrig.maya import util, lib, draw

# kCmdName = "EphRigContextCmdName"
# kContextName = "EphRigContextName"

maya_useNewAPI = True



class EphRigMpxAnimContext(omui.MPxContext):
	"""main context for ephrig manipulation"""

	kToolName = "EphRigAnimCtx"

	def __init__(self):
		#super(EphRigMpxAnimContext, self).__init__()
		omui.MPxContext.__init__(self)
		self.setTitleString("EphRig Context")
		print("MPxContext init")

		self.rigs = set()
		self.gatherRigs()

	def gatherRigs(self):
		"""find all valid EphRig nodes in scene,
		reconstruct rigs from them"""
		rigNodes = cmds.ls(type="ephRigMain") or []
		rigs = [util.getRigFromNode(i) for i in rigNodes]
		self.rigs = set(rigs)
		#print("gathered rigs", self.rigs)



	def __del__(self):
		print("del context")

	@classmethod
	def creator(cls):
		print("context creator")
		return EphRigMpxAnimContext()


	def drawFeedback(self,
	                 #event,
	                 drawMgr, context):
		"""
		:param event: ??
		:param drawMgr: omui.MUIDrawManager
		:param context: omui.MFrameContext
		:return:
		"""
		# print("ephrig ctx draw feedback")
		# print(drawMgr, context)
		for rig in self.rigs:
			draw.drawRig(rig, drawMgr)




	def stringClassName(self):
		"""returning custom name crashes immediately,
		even with correct mel files set up"""
		# print("ctx string class name")
		base = super(EphRigMpxAnimContext, self).stringClassName()
		# print("super", base)
		return base
		#return self.kToolName

	def toolOnSetup(self):
		print("ephRigMpxContext setup")
		self.setHelpString("EphRig context active :D")
		self.gatherRigs()

	#@classmethod
	def toolOffCleanup(self):
		print("ephRigMpxContext cleanup")
		pass


class EphRigContextSetupCmd(omui.MPxContextCommand):
	kPluginCmdName = "EphNewRigContextCmdName"

	def __init__(self):
		print("MpxContexCmd init")
		# super(EphRigContextSetupCmd, self).__init__()
		omui.MPxContextCommand.__init__(self)

	@classmethod
	def creator(cls):
		print("EphRigContext cmd creator")
		return cls()

	def makeObj(self):
		print("context cmd makeObj")
		return EphRigMpxAnimContext()

	def __del__(self):
		print("del context cmd")


def updateMelFiles(baseDir, toolName):
	"""tool contexts apparently freak out if
	<Tool name>Properties.mel and
	<Tool name>Values.mel aren't found
	gotta love it

	:param baseDir: dir in which to replace or generate
	stub mel files
	:param toolName: final name of tool. Must not
	contain spaces
	"""

	for i in ("Properties", "Values"):
		fnName = f"{toolName}{i}"
		fileName = fnName + ".mel"
		path = os.path.join(baseDir, fileName)
		print("fp", path)


		if i == "Properties":
			procString = f"global proc {fnName}()" + "{}"
		else:
			procString = f"global proc {fnName}(string $toolName)" + "{}"
		with open(path, "w") as f:
			f.write(procString)
