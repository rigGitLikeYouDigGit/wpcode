
from __future__ import annotations
import typing as T


"""
package for python maya plugins -
this can be hard-reloaded independently from the main
wp packages

"""

from wplib import WP_ROOT_PATH
from wpm import WN
from wpm.lib.plugin import PluginNodeTemplate, WpPluginAid

from wpmplugin.testNode import TestNode

thisFilePath = WP_ROOT_PATH / "code" / "maya" / "plugin" / "wpmplugin" / "__init__.py"

# construct overall plugin object, register any python plugin nodes
wpPyPlugin = WpPluginAid(
	"wpPy",
	pluginPath=str(thisFilePath),
	nodeClasses={
		1 : TestNode
	}
)


def maya_useNewAPI():
	pass


def initializePlugin(plugin):
	"""initialise the plugin"""
	wpPyPlugin.initialisePlugin(plugin)

def uninitializePlugin(plugin):
	"""uninitialise the plugin"""
	wpPyPlugin.uninitialisePlugin(plugin)











