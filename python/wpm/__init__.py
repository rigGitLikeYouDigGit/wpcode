
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

from pathlib import Path
MAYA_PY_ROOT = Path(__file__).parent

from wplib import to, arr

def isThisMaya():
	try:
		from maya import cmds
		return True
	except (ModuleNotFoundError, ImportError):
		return False


def findKey(namespace, value):
	"""put this somewhere
	find the constant name for a given value in a given namespace
	"""
	vMap = {}
	for k, v in namespace.__dict__.items():
		try:
			vMap[v] = k
		except:
			pass
	return vMap.get(value)



if isThisMaya():
	from .core import (
		cmds, om, oma, omui, omr, # wrapped maya modules
		getCache, # api cache
		#WN, createWN, # node wrappers
		WN,
		Plug,
		getSceneGlobals, # scene wrappers
		api,
	getMFn, getMPlug, getMObject, getMDagPath, getMFnType,
	use


	)

	# use to denote function arguments to accept either static values or plugs
	PLUG_VAL = (Plug, object)
