
from __future__ import  annotations
import types, typing as T

import builtins
from pathlib import Path
import inspect, importlib, pkgutil

from dataclasses import dataclass

from wplib import log


"""
run over all given modules packages to find functions 
to expose as nodes
"""

def getModules(rootNames=(),
                 rootPaths=(),
                 recurse=False)->dict[str, types.ModuleType]:
	"""probably want to set this as async or threaded, since it might take a while"""
	# save the python import path against module object
	modules : dict[str, types.ModuleType] = {}
	for i in rootNames:
		try:
			module = importlib.import_module(i)
		except ImportError:
			log(f"could not import module path {i}, skipping")
			continue

		log("found module", module)
		if recurse:
			for moduleInfo in pkgutil.walk_packages(module.__path__, prefix=module.__name__ + "."):
				try:
					modules[moduleInfo.name] = importlib.import_module(moduleInfo.name)
				except ImportError:
					log(f"could not import module {moduleInfo}, skipping")
					continue
	return modules

def moduleItems(module:types.ModuleType):
	"""do we allow pulling in constants too"""
	ms = inspect.getmembers(module)






def callablesForType(t:type):
	"""given a type, return all functions that can be
	given as info"""



if __name__ == '__main__':
	result = getModules(("wplib", ))
	for k, v in result.items():
		print(k, v)


