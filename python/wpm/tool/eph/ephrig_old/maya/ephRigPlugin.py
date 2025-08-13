
import sys, os
from importlib import reload
from maya import cmds
from maya.api import OpenMaya as om, OpenMayaRender as omr

from edRig.ephrig.maya import pluginnode, plugindrawoverride
#reload(pluginnode)
EphRigControlNode = pluginnode.EphRigControlNode

from edRig.ephrig.maya import plugincontext
#reload(plugincontext)

maya_useNewAPI = True



def initializePlugin(mobject):
	print("eph plugin init")
	pluginFn = om.MFnPlugin(mobject)
	#pluginFn.setName("ephRigPlugin") # setName crashes maya lol
	# unfortunately the name of the python file is also the name of the plugin

	pluginFn.registerNode(
		EphRigControlNode.typeName,
		EphRigControlNode.typeId,
		EphRigControlNode.creator,
		EphRigControlNode.initialize,
		om.MPxNode.kLocatorNode,
		EphRigControlNode.drawClassification
	)

	omr.MDrawRegistry.registerDrawOverrideCreator(
		EphRigControlNode.drawClassification,
		EphRigControlNode.drawRegistrantId,
		plugindrawoverride.EphRigDrawOverride.creator
	)

	pluginFn.registerContextCommand(
		plugincontext.EphRigContextSetupCmd.kPluginCmdName,
		plugincontext.EphRigContextSetupCmd.creator
	)

def uninitializePlugin( mobject ):
	print("eph uninit")
	pluginFn = om.MFnPlugin(mobject)

	omr.MDrawRegistry.deregisterDrawOverrideCreator(
		EphRigControlNode.drawClassification,
		EphRigControlNode.drawRegistrantId
	)

	pluginFn.deregisterNode(EphRigControlNode.typeId)

	pluginFn.deregisterContextCommand(
		#plugincontext.kContextName,
		plugincontext.EphRigContextSetupCmd.kPluginCmdName,
	)
