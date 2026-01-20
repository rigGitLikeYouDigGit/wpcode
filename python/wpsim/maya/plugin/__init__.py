from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm.lib.plugin import PluginNodeTemplate, MayaPyPluginAid

from wpsim.maya.plugin.rigidbody import WpSimRigidBodyNode, WpSimBodyMPxData


"""combined plugin system for various simulation systems
in wpsim
"""

wpSimPlugin = MayaPyPluginAid(
	name="wpsim",
	pluginPath=__file__,
	nodeClasses={
		1 : WpSimRigidBodyNode
	},
	mpxDataClasses={
		34 : WpSimBodyMPxData
	}

)

