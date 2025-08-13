from maya import cmds
from maya.api import OpenMaya as om

from edRig.maya.lib.plugin import _registerNode, _deregisterNode, registerLoadedPluginName, registeredClses, _registerDrawOverride, _deregisterDrawOverride

from edRig.maya.tool.feldspar.plugin.constant import *
from edRig.maya.tool.feldspar.plugin.solvernode import FeldsparSolverNode

from edRig.maya.tool.feldspar.plugin.setupnode import FeldsparSetupNode

from edRig.maya.tool.feldspar.plugin.drawoverride import FeldsparDrawOverride


def maya_useNewAPI():
	pass


def initializePlugin(mobject):
	mplugin = om.MFnPlugin(mobject)
	_registerNode(mplugin, FeldsparSetupNode)
	_registerNode(mplugin, FeldsparSolverNode)

	_registerDrawOverride(mobject,
	                      FeldsparDrawOverride,
	                      forNodeCls=FeldsparSolverNode)


def uninitializePlugin( mobject ):
	mPlugin = om.MFnPlugin(mobject)
	for nodeCls in registeredClses:
		_deregisterNode(mPlugin, nodeCls)
	_deregisterDrawOverride(FeldsparDrawOverride,
	                        forNodeCls=FeldsparSolverNode)


# pluginPath = __file__
pluginPath = r"F:\all_projects_desktop\common\edCode\edRig\maya\tool\feldspar\plugin\__init__.py"
def loadPlugin():
	cmds.loadPlugin(pluginPath, name=PLUGIN_NAME)
	registerLoadedPluginName(PLUGIN_NAME)

def unloadPlugin():
	#cmds.unloadPlugin(pluginPath, force=1)
	cmds.unloadPlugin(PLUGIN_NAME, force=1)
