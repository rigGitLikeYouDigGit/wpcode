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
in wpsim.

we allow users to define their own functions for use in constraint building -

simResource node loads set of python modules, runs user-defined functions
for constraints and measurements - that node connects to each sim node
using its functions? might be a lot of connections but it's fine.
resource node namespaces overridden in order their connections are specified
 

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

