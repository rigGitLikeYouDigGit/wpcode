from __future__ import annotations
import typing as T

import os, sys
from pathlib import Path

"""very, very dodgy - we use the old maya api for this node alone,
as it's the only place that MPxObjectSet is exposed to python.
We do no direct computation with it, only setting up callbacks
when the node is constructed.

This file should be self-contained as the entire plugin
"""
import maya.OpenMayaMPx as ompx
import maya.OpenMaya as omOld



from wpm.lib.plugin import MayaPluginAid, NodeParamData, PluginNodeTemplate


class ActiveSetNode(ompx.MPxObjectSet, PluginNodeTemplate):
	kNodeName = "wpActiveSet"
	kNodeClassify = "utility/general"
	#kNodeTypeId = omOld.MTypeId(0x00000001)
	kNodeLocalId = 0

	pass


# set up wrappers for plugin registration
def getThisFilePath():
	"""annoying replacement for __file__, since registering
	a plugin directly executes this python file (but doesn't correctly
	add it to sys.argv)
	"""
	from wpm import plugin
	return Path(plugin.__file__).parent / "activeset.py"



pluginAid = MayaPluginAid(
	name="wpActiveSetNode",
	studioLocalPluginId=1,
	pluginPath=getThisFilePath(),
	nodeClasses=(ActiveSetNode,),
	drawOverrideClasses={},
)

def initializePlugin(mobject):
	pluginAid.initialisePlugin(mobject, ompx.MFnPlugin)

def uninitializePlugin(mobject):
	pluginAid.uninitialisePlugin(mobject, ompx.MFnPlugin)



