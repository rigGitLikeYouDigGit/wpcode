from __future__ import annotations
import types, typing as T
import pprint
#from wplib import log

from wplib.sentinel import Sentinel

unrollT = T.TypeVar("unrollT")
def unroll(iterable:T.Iterable[unrollT],
           nextTargetsFn:T.Callable[
	           [unrollT, ... ], T.Iterable[unrollT]],
           *args, **kwargs)->T.Iterable:
	"""unroll an otherwise recursive iteration by using a stack to keep track of next targets"""
	toCheck = list(iterable)
	while toCheck:
		target = toCheck.pop(0)
		yield target
		toCheck.extend(nextTargetsFn(target, *args, **kwargs))


class OverrideProvider:
	"""Mixin to allow pinning values at points
	in a hierarchy, affecting all elements below.
	Important to only traverse when value changed,
	otherwise directly copy parent when new object created.
	so this operates on a push system, not pull

	TODO: internal override map could of course be a dict or tree,
		any container; resolving that in turn could make use of overrides,
		so it's all the same really

	we just use single-layer dict for now

	_explicitOverrides : overrides defined on this object
	_overrideCache : cache of override values, including explicit
		on this object
	"""

	def __init__(self, defaultOverrides:dict=None):
		self._explicitOverrides = {}
		self._overrideCache = {}

	def _getExplicitOverrides(self)->dict:
		"""
		return a live dict of all overrides set directly
		on this object"""
		return self._explicitOverrides

	def recacheChildren(self, baseCache:dict=None, _seenCache:set=None):
		"""clear the cache of all children, so they will recheck for overrides on next access
		kind of has to be recursive here since we have to revert the state
		of the cache on each branch
		"""
		seenCache = set() if _seenCache is None else _seenCache
		if baseCache is None:
			baseCache = self._overrideCache
		baseCache.update(self._explicitOverrides)
		self._overrideCache = dict(baseCache)
		for k, v in baseCache.items():
			childrenIter = self._getOverrideChildren(forKey=k)
			for child in childrenIter:
				if hasattr(child, "__hash__") and child in seenCache:
						continue
				seenCache.add(child)
				child.recacheChildren(baseCache)

	def updateOverride(self, data:dict, recacheChildren=True):
		"""set an override value on this object"""
		self._getExplicitOverrides().update(data)
		if recacheChildren:
			self.recacheChildren()

	def _getOverrideParents(self, forKey="")->T.Iterable[OverrideProvider]:
		"""return the direct ancestor(s) of this
		provider to look at for the given key"""
		raise NotImplementedError(self)

	def _getOverrideChildren(self, forKey="")->T.Iterable[OverrideProvider]:
		"""return the direct child(ren) of this
		provider to look at for the given key"""
		raise NotImplementedError(self)

	def getOverride(self, key:str, default=Sentinel.FailToFind,):
		"""return the override value for the given key, checking this object and then ancestors
		"""
		found = self._overrideCache.get(key, default)
		if found is Sentinel.FailToFind:
			raise KeyError
		return found

	def getOverrideProvider(self, key:str, onlyFirst=True)->list[tuple[
		OverrideProvider, T.Any]]:
		""" return an override value for all this object's
		override ancestors
		if obj, return the provider object with the given override
		set
		if not onlyFirst, exhaust every accessible override provider
		to find overrides set against the given key, and return them in a list of tuples:
		[ (provider object, override value) ]

		TODO: make this an iterator on returning multiple values,
			so external processes can check returned overrides and decide outside if they
			need to continue
		"""

		toCheck = [self]
		found = []
		while toCheck:
			provider = toCheck.pop(0)
			overrides = provider._getExplicitOverrides()
			result = overrides.get(key, default=Sentinel.FailToFind)
			if result is Sentinel.FailToFind: # on miss, go looking up on ancestors
				toCheck.extend(provider._getOverrideParents(forKey=key))
				continue
			found.append((provider, result))
			if onlyFirst:
				return found
			toCheck.extend(provider._getOverrideParents(forKey=key))
		return found

		# 	# check if we want object or value
		# 	if returnValue and returnProvider: # fizzbuzz pros in shambles
		# 		found.append((result, provider))
		# 	elif returnProvider:
		# 		found.append(provider)
		# 	elif returnValue:
		# 		found.append(result)
		# 	break
		#
		# if not found: # nothing found
		# 	if not onlyFirst: # return empty list
		# 		return found
		# 	if default is Sentinel.FailToFind:
		# 		raise KeyError(f"No override found for key {key} on object {self} ")
		# 	return default


# test for a more flexible approach looking for
# arbitrary attributes on objects
# (for example Qt widgets)
def _getDirectOverrides(obj, attributeDictName, default):
	if isinstance(obj, OverrideProvider):
		return obj._getExplicitOverrides()
	return getattr(obj, attributeDictName, default)
def getOverride(obj, key: str,
                attributeDictName: str,
                getAncestorsFn:T.Callable[[T.Any], tuple[T.Any]],
				default=Sentinel.FailToFind,
                returnObj=False,
                onlyFirst=True,
                ):
	"""TODO: we don't need a whole other system of adaptors for this -
			maybe could extend visitAdaptor
	"""
	toCheck = [obj]
	found = []
	while toCheck:
		provider = toCheck.pop(0)
		#overrides = provider._getDirectOverrides()
		#ancestors = provider._getOverrideAncestors(forKey=key) if isinstance(provider, OverrideProvider) else getAncestorsFn(obj)
		ancestors = getAncestorsFn(obj)
		overrides = _getDirectOverrides(obj, attributeDictName, {})
		result = overrides.get(key, default=Sentinel.FailToFind)
		if result is Sentinel.FailToFind:  # on miss, go looking up on ancestors
			#toCheck.extend(provider._getOverrideAncestors(forKey=key))
			toCheck.extend(ancestors)
			continue
		# check if we want all matching overrides
		if not onlyFirst:
			found.append((provider, result))
			toCheck.extend(ancestors)
			continue
		# check if we want object or value
		if returnObj:
			found.append(provider)
		else:
			found.append(result)
		break

	if not found:  # nothing found
		if not onlyFirst:  # return empty list
			return found
		if default is Sentinel.FailToFind:
			raise KeyError(f"No override found for key {key} on object {self} ")
		return default

	return found[0]