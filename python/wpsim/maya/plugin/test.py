from __future__ import annotations
import types, typing as T
import pprint
from importlib import reload


import wpm

from wpm.core.node import base

from wpm import om, cmds, WN

# from wpsim.maya import plugin
# from wpsim.maya.plugin import rigidbody
from wpm.core.node.codegen import gather, generate

def runTest():
	# gather.updateDataForNodes()
	#generate.genNodes()
	from wpsim.maya import plugin
	from wpsim.maya.plugin import rigidbody

	reload(rigidbody)
	reload(plugin)
	cmds.file(new=1, f=1)
	plugin.wpSimPlugin.unloadPlugin()




	plugin.wpSimPlugin.loadPlugin(forceReload=True)
	WN.syncWrappersForPlugin(plugin.wpSimPlugin.name)
	body = WN.createNode("wpSimRigidBody")

	WN.Mesh.antialiasingLevel_
	print(body, type(body))




