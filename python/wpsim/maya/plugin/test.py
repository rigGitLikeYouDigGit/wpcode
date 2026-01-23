from __future__ import annotations
import types, typing as T
import pprint
from importlib import reload


import wpm

from wpm.core.node import base

from wpm import om, cmds, WN

from wpsim.maya import plugin


def runTest():

	cmds.file(new=1, f=1)
	plugin.wpSimPlugin.loadPlugin(forceReload=True)
	body = WN.createNode("wpSimRigidBody")
	print(body, type(body))




