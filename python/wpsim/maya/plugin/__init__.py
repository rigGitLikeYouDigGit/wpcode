from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from importlib import reload
from wpm.lib.plugin import PluginNodeTemplate, MayaPyPluginAid


from wpsim.maya.plugin import rigidbody, forcefield
reload(rigidbody)
reload(forcefield)
#from wpsim.maya.plugin.rigidbody import WpSimRigidBodyNode, WpSimBodyMPxData


"""combined plugin system for various simulation systems
in wpsim
"""


wpSimPlugin = MayaPyPluginAid(
	name="wpsim",
	pluginPath="C:/Users/ed/Documents/GitHub/wpcode/python/wpsim/maya/plugin"
	           "/pluginMain.py",
	nodeClasses={
		1 : rigidbody.WpSimRigidBodyNode,
		2 : forcefield.WpSimForceFieldNode,
	},
	drawOverrideClasses={
		forcefield.WpSimForceFieldNode: forcefield.WpSimForceFieldDrawOverride
	},
	mpxDataClasses={
		34 : rigidbody.WpSimBodyMPxData,
		35 : forcefield.WpSimForceFieldMPxData,
	}

)

