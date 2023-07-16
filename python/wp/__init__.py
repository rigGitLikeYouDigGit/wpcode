
from __future__ import annotations
"""any standalone code for tools or pipeline goes here
maybe for this package, we just assume that everything is a lib
unless specified otherwise

depends on wplib and wptree, but not on any other wp packages
"""

import sys, gc
def reloadWP():
	"""reloads the wpm module and all submodules"""
	for i in tuple(sys.modules.keys()):
		if i.startswith("wp"):
			del sys.modules[i]
	gc.collect(2)
	import wp
	return wp


from . import env

