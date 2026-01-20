from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


from wpsim.maya.plugin import wpSimPlugin


def initializePlugin(mobject):
	wpSimPlugin.initialisePlugin(mobject)

def uninitializePlugin(mobject):
	wpSimPlugin.uninitialisePlugin(mobject)
