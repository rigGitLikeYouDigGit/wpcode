
from __future__ import annotations
"""
Module holding the patched cmds module

This module is imported by __init__ first, loading it
and setting up references to normal cmds, om -

we then patch them into their WP versions

"""

import typing as T
import pathlib, importlib, sys, types


def directModuleImport(modulePath:str)->types.ModuleType:
	"""directly loads a module without running init"""

	"""Import a module from the given path."""
	module_path = pathlib.Path(modulePath).resolve()
	module_name = module_path.stem  # 'path/x.py' -> 'x'
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module

# vanilla maya modules, copied so none of our jank leaks back into maya itself
baseCmds = importlib.import_module("maya.cmds")
baseOm = importlib.import_module("maya.api.OpenMaya")
baseOmr = importlib.import_module("maya.api.OpenMayaRender")
baseOma = importlib.import_module("maya.api.OpenMayaAnim")
baseOmui = importlib.import_module("maya.api.OpenMayaUI")

# wp versions - these can be renamed when imported at higher scope
# more helpful to be explicit here
wpCmds = baseCmds
wpOm = baseOm
wpOmr = baseOmr
wpOma = baseOma
wpOmui = baseOmui