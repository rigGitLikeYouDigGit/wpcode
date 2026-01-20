from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm import om, cmds, WN

from wpsim.maya.plugin import wpSimPlugin

def runTest():
	cmds.file(new=1, f=1)
	wpSimPlugin.loadPlugin(forceReload=True)
	body = WN.createNode("wpSimRigidBody")




