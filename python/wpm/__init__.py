
"""maya code

tried a smart guard in the past to allow code that imported openmaya, blender and houdini
in the same file - worked, but never gave much advantage. Can implement something similar
if needed


we will use some core "smart" objects to make it easier to work with maya - presents
issue of module structure and dependency.
core : functions interfacing directly with api
object : smart objects. These should be small, atomic - depend on core, but not on lib?

lib : anonymous library functions, may depend on objects


tool : discrete tools, may depend on lib

"""

import sys, gc
def reloadWPM(andWp=True):
	"""reloads the wpm module and all submodules
	optionally also reload all of wp"""
	from wp import reloadWP
	if andWp:
		wp = reloadWP()
	for i in tuple(sys.modules.keys()):
		if i.startswith("wpm"):
			del sys.modules[i]
		if i.startswith("tree"):
			del sys.modules[i]
	gc.collect(2)
	import wpm
	return wpm


def isThisMaya():
	try:
		from maya import cmds
		return True
	except ModuleNotFoundError:
		return False

if isThisMaya():
	from .core import (
		cmds, om, oma, omui, omr, # wrapped maya modules
		getCache, # api cache
		#WN, createWN, # node wrappers
		WN,
		Plug,
		getSceneGlobals, # scene wrappers
		api,
	getMFn, getMPlug, getMObject, getMFnType

	)

	# use to denote function arguments to accept either static values or plugs
	PLUG_VAL = (Plug, object)
