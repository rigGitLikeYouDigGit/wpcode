from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from importlib import reload
from wpm.lib.plugin import PluginNodeTemplate, MayaPyPluginAid


from wpsim.maya.plugin import rigidbody
reload(rigidbody)
#from wpsim.maya.plugin.rigidbody import WpSimRigidBodyNode, WpSimBodyMPxData


"""combined plugin system for various simulation systems
in wpsim
"""


wpSimPlugin = MayaPyPluginAid(
	name="wpsim",
	pluginPath="C:/Users/ed/Documents/GitHub/wpcode/python/wpsim/maya/plugin"
	           "/pluginMain.py",
	nodeClasses={
		1 : rigidbody.WpSimRigidBodyNode
	},
	mpxDataClasses={
		34 : rigidbody.WpSimBodyMPxData
	}

)

