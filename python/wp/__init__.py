
from __future__ import annotations
"""
any standalone code for tools or pipeline goes here
maybe for this package, we just assume that everything is a lib
unless specified otherwise

depends on wplib and wptree and chimaera, but not on any other wp packages

WP NAME CLASHES WITH WARP BY NVIDIA
OH NO

for domain-specific pipeline work, where to put it?
wp.maya?
wpm.pipe?

wp.maya, pure wpm code shouldn't explicitly depend on pipe?
not sure where chimaera and the rig parts fit in
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
from .pipe import Asset, Show
