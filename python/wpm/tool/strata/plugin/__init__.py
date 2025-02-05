
from __future__ import annotations
import typing as T


"""
"""

from wplib import WP_ROOT_PATH
from wpm import WN
from wpm.lib.plugin import PluginNodeTemplate, WpPluginAid

from wpm.tool.strata.plugin.point import StrataPoint

from wpm.tool import strata

from pathlib import Path
thisFilePath = Path(strata.__file__).parent / "plugin" / "__init__.py"

# class StrataPluginAid(MayaPluginAid):
# 	@classmethod
# 	def studioIdPrefix(cls)->int:
# 		"""return the prefix for studio plugin ids,
# 		used as a namespace for plugin node ids"""
# 		return WP_PLUGIN_NAMESPACE

#F:\wp\code\maya\plugin\strata\Debug
# construct overall plugin object, register any python plugin nodes
pluginAid = WpPluginAid(
	"strata",
	pluginPath=str(WP_ROOT_PATH/"code"/"maya"/"plugin"/"strata"/"Debug"/"strata.mll"),
	# nodeClasses={
	# 	1 : StrataPoint
	# }
)


def maya_useNewAPI():
	pass


def initializePlugin(plugin):
	"""initialise the plugin"""
	pluginAid.initialisePlugin(plugin)

def uninitializePlugin(plugin):
	"""uninitialise the plugin"""
	pluginAid.uninitialisePlugin(plugin)

def reloadPluginTest():
	"""single test to check strata nodes are working"""
	from maya import cmds
	cmds.file(newFile=1, f=1)
	pluginAid.loadPlugin(forceReload=True)
	pt = cmds.createNode("strataPoint")
	# parentLoc = cmds.spaceLocator(name="parentLoc")[0]
	# cmds.setAttr(parentLoc + ".translate", 2, 3, 4)

	# cmds.connectAttr(parentLoc + ".worldMatrix[0]", pt + ".parent[0].parentMatrix")
	#
	# childLoc = cmds.spaceLocator(name="childLoc")[0]
	# cmds.connectAttr(pt + ".outMatrix", childLoc + ".parentOffsetMatrix")


	# StrataPoint.testNode()











