from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
import importlib
import jax
from jax import jit, vmap, grad, numpy as jnp

"""probably don't need these to be lib functions
"""
def loadUserModule(moduleName: str):
	"""Load a user module by name."""
	module = importlib.import_module(moduleName)
	return module

def updateNamespace(ns: dict, module: types.ModuleType):
	"""Update a namespace with the contents of a module."""
	for name in dir(module):
		ns[name] = getattr(module, name)
