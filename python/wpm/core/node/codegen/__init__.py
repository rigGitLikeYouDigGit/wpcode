
from __future__ import annotations
import typing as T
from types import ModuleType

import importlib as _importlib

if T.TYPE_CHECKING:
	from .gen import *
	from .modified import *
	pass

"""tests for generating node wrappers procedurally
from maya's own nodes

import modules from this module

this top-level init unifies modified and source packages - 
no direct relation to the maya system, can be reused for other uses of codegen

"""


_moduleCache : dict[str : ModuleType] = {}

def clearModuleCache():
	_moduleCache.clear()

def __getattr__(name):
	try:
		return _moduleCache[name]
	except KeyError:
		pass

	# get module
	module = None
	try:
		module = _importlib.import_module(f"wpm.core.node.codegen.modified.{name}")
	except ImportError:
		pass
	if module is None:
		try:
			module = _importlib.import_module(f"wpm.core.node.codegen.gen.{name}")
		except ImportError:
			pass

	if module is None:
		raise ImportError("Unknown lookup for codegen module", name)
	_moduleCache[name] = module
	return module



	#


