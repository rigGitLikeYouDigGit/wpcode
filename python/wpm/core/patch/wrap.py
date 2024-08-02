"""Module for wrapping cmds at startup,
and holding conversion decorators to show which functions
accept which types

also patches __getattribute__ on cmds module, so that
cmds functions are monkeypatched lazily on lookup
"""
from __future__ import annotations
from typing import List, Set, Dict, Callable, Tuple, Sequence, Union, TYPE_CHECKING
from functools import partial, wraps
import sys, traceback, types, pathlib, importlib
from pathlib import Path


# import the node base / mixin for instance checking
#from . import bases as mod
#reload(mod)
from wpm.core.bases import NodeBase, PlugBase


# let sets be hashable
import forbiddenfruit
hashLambda = lambda obj: id(obj)
forbiddenfruit.curse(set, "__hash__", hashLambda)

baseSet = {"a", "b"}
outerSet = {baseSet, 24}

def addHashFnToCls(cls):
	# om.MObject.__slots__ = tuple(list(om.MObject.__slots__)
	#                              + ["__weakref__"])
	cls.__hash__ = hashLambda
	pass



def patchMObjectHash(omModule:types.ModuleType):
	"""you need to pull my specific fork of forbiddenfruit for this to work,
	the base package doesn't handle __hash__ properly

	give MObjects a consistent __hash__ using the builtin MObjectHandle,
	letting them work in dicts, with weakrefs, etc
	"""
	import forbiddenfruit
	objHash = lambda obj: omModule.MObjectHandle(obj).hashCode()
	forbiddenfruit.curse(omModule.MObject, "__hash__", objHash)

def patchMPlugHash(omModule:types.ModuleType):
	"""
	"""
	import forbiddenfruit
	objHash = lambda plug: hash((plug.node(), plug.attribute(), plug.logicalIndex())
	                            if plug.isElement() else (plug.node(), plug.attribute()))
	forbiddenfruit.curse(omModule.MPlug, "__hash__", objHash)



def returnList(wrapFn):
	"""I love it when cmds functions can return
	either a list or None :)"""
	@wraps(wrapFn)
	def _innerFn(*args, **kwargs):
		result = wrapFn(*args, **kwargs)
		if result is None:
			#print("returned None, changing to list")
			return []
		return result

	return _innerFn


listFunctions = ["ls", "listRelatives", "listHistory", "listConnections",
	                 "listAttr"]

def isInstanceReloadSafe(obj, checkType):
	"""AWFUL solution and I'm ashamed to need it
	reloading in python is just such a greaaaat system
	type checking robust to reloading obj / checkType module
	taking hash of the type doesn't work
	"""
	return str(checkType) in (str(i) for i in type(obj).__mro__)


def wrapListFns():
	# patch maya cmds "list-" functions to return lists no matter what
	for fnName in listFunctions:
		try:
			fn = getattr(cmds, fnName)
			# check if has already run
			base = getattr(cmds, "_" + fnName, None)
			if base:
				setattr(cmds, fnName, returnList(base))
			else:
				# set original function to "_fnName"
				setattr(cmds, "_" + fnName, fn)
				# update module refrence with wrapped
				setattr(cmds, fnName, returnList(fn))

		except:
			print(("error wrapping {}".format(fnName)))
			print((traceback.format_exc()))

def typeSwitchInPlace(obj, typeMap:Dict[type, Callable], copy=False):
	"""given arbitrary prim structure, either return flat converted object
	or replace objects in place"""

	if isinstance(obj, (tuple, list, set)):
		newData = [typeSwitchInPlace(i, typeMap) for i in obj]
		return type(obj)(newData)
	elif isinstance(obj, (dict, )):
		newKeys = [typeSwitchInPlace(i, typeMap)
		           for i in obj.keys()]
		newValues = [typeSwitchInPlace(i, typeMap)
		             for i in obj.values()]
		return {k : v for k, v in zip(newKeys, newValues)}

	# is singular object, convert
	# basic iteration here, we don't expect hugely rich type mappings on functions
	for srcType, dstType in typeMap.items():
		strParity = isInstanceReloadSafe(obj, srcType)
		if isinstance(obj, srcType): # take first match
			try:
				return dstType(obj)
			except:
				print("direct conversion encountered issue with",
				      obj, type(obj))
				continue
	return obj


# actual function to wrap arbitrary functinos
def typeSwitchParamsPatch(fn, typeMap):
	"""probably not consistent to have functions return the
	same type as passed in - would need more complex logic"""
	#@wraps(fn)
	def wrapper(*fnArgs, **fnKwargs):
		"""convert argument types"""
		newArgs = typeSwitchInPlace(fnArgs, typeMap)
		newKwargs = typeSwitchInPlace(fnKwargs, typeMap)
		result = fn(*newArgs, **newKwargs)
		return result
	return wrapper

def wrapCmdFn(fn):
	"""wrap single cmd to flatten NodeBases (AbsoluteNodes)
	to strings when passed as params"""
	# print("wrapping cmd", fn.__name__)
	fnName = fn.__name__
	typeMap = {NodeBase : str,
	           PlugBase : str,
	           set : list} # can add entries for pm nodes, cmdx etc
	fn = typeSwitchParamsPatch(fn, typeMap)

	# add list return patch
	if fnName in listFunctions:
		fn = returnList(fn)
	return fn

wrappedCache = {}

def _cmdsGetAttribute_(module, name):
	"""method to supplant normal __getattribute__
	on the cmds module
	use to wrap cmds functions lazily as needed
	functions cached in dict for now,
	not explicitly set back on module yet

	we also do not yet enforce returning EdNodes for
	cmds results - that may come in time, but it feels
	too much like PyMel
	"""

	# check cache for wrapped function
	found = wrappedCache.get(name, None)
	if found: return found

	# no wrapped exists - retrieve it, wrap it, cache it
	found = object.__getattribute__(module, name)

	# if it's not a function, just return it
	if not callable(found):
		return found

	patchFn = wrapCmdFn(found)
	wrappedCache[name] = patchFn
	return patchFn



# whitelist any problematic cmds


# this is a way simpler idea
class ModuleDescriptor(object):
	"""wrapper class for a module to implement
	accessors easily"""
	def __init__(self, module):
		self.mod = module

	def __getattribute__(self, item):
		mod = object.__getattribute__(self, "mod")
		return getattr(mod, item)

class CmdsDescriptor(ModuleDescriptor):

	def __getattribute__(self, item):
		if item == "mod" :
			return object.__getattribute__(self, "mod")
		return _cmdsGetAttribute_(self.mod, item)


def wrapCmds(targetModule:types.ModuleType,
             baseMemberName:str,
             resultMemberName:str):
	"""pass in target host module,
	with existing reference to normal god-fearing cmds module
	corrupt it with our degeneracy
	"""
	# wrap list functions to return lists
	# implement lazy lookup cache to flatten suitable arguments to strings
	print("wrapping cmds")

	cmds = getattr(targetModule, baseMemberName)
	descriptor = CmdsDescriptor(cmds)
	setattr(targetModule, resultMemberName, descriptor)


def wrapOm(targetModule:types.ModuleType,
            baseMemberName:str,
           resultMemberName:str):
	"""pass in target host module,
	patch up openmaya module"""

	baseOm = getattr(targetModule, baseMemberName)
	patchMObjectHash(baseOm)
	patchMPlugHash(baseOm)

	setattr(targetModule, resultMemberName, baseOm)




