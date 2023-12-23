


from __future__ import annotations
import typing as T

import os, sys
from pathlib import Path


from wpm.lib.plugin import MayaPluginAid, NodeParamData, PluginNodeTemplate

from wpm.plugin.activeset.node import ActiveSetNode


"""plugin object set node, actively updating its own membership
based on expressions.
Relies on listening to scene messages, and exposes a 'balanceWheel'
bool attribute that flips when the set membership changes.

To connect this to further systems, just listen to the balanceWheel attribute.

"""

# set up wrappers for plugin registration
def getThisFilePath():
	"""annoying replacement for __file__, since registering
	a plugin directly executes this python file (but doesn't correctly
	add it to sys.argv)
	"""
	from wpm.plugin import activeset
	return Path(activeset.__file__)



pluginAid = MayaPluginAid(
	name="wpActiveSetNode",
	studioLocalPluginId=1,
	pluginPath=getThisFilePath(),
	nodeClasses=(ActiveSetNode,),
	drawOverrideClasses={},
	useOldApi=True
)

def initializePlugin(mobject):
	pluginAid.initialisePluginOldApi(mobject)

def uninitializePlugin(mobject):
	pluginAid.uninitialisePluginOldApi(mobject)



