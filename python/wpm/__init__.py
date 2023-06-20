
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
import sys
def reloadWPM():
	"""reloads the wpm module and all submodules"""
	for i in tuple(sys.modules.keys()):
		if i.startswith("wpm"):
			del sys.modules[i]
	import wpm
	return wpm

from .core import (
	cmds, om, oma, omui, omr, # wrapped maya modules
	getCache, # api cache
	WN, createWN, # node wrappers
	getSceneGlobals, # scene wrappers

)


