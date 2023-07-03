
"""houdini code.

Unsure of how to split python code between tools and HDAs -
for now each worthy HDA gets its own tool package
"""


import sys, gc
def reloadWPH(andWp=True):
	"""reloads the wpm module and all submodules
	optionally also reload all of wp"""
	from wp import reloadWP
	if andWp:
		wp = reloadWP()
	for i in tuple(sys.modules.keys()):
		if i.startswith("wph"):
			del sys.modules[i]
		if i.startswith("tree"):
			del sys.modules[i]
	gc.collect(2)
	import wph
	return wph


