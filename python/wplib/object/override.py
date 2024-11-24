from __future__ import annotations
import types, typing as T
import pprint
#from wplib import log

from wplib.sentinel import Sentinel

class OverrideProvider:
	"""Mixin to allow pinning values at points
	in a hierarchy, affecting all elements below
	"""

	def __init__(self):
		self._overrides = {}

	def _getDirectOverrides(self)->dict:
		"""
		return a live dict of all overrides set directly
		on this object"""
		return self._overrides

	def setOverride(self, key:str, value):
		"""set an override value on this object"""
		self._getDirectOverrides()[key] = value

	def _getOverrideAncestors(self, forKey="")->T.Iterable[OverrideProvider]:
		"""return the direct ancestor(s) of this
		provider to look at for the given key"""
		raise NotImplementedError(self)

	def getOverride(self, key:str, default=Sentinel.FailToFind,
	                 returnObj=False,
	                onlyFirst=True):
		""" return an override value for all this object's
		override ancestors
		if obj, return the provider object with the given override
		set
		if not onlyFirst, exhaust every accessible override provider
		to find overrides set against the given key, and return them in a list of tuples:
		[ (provider object, override value) ]
		"""
		toCheck = [self]
		found = []
		while toCheck:
			provider = toCheck.pop(0)
			overrides = provider._getDirectOverrides()
			result = overrides.get(key, default=Sentinel.FailToFind)
			if result is Sentinel.FailToFind: # on miss, go looking up on ancestors
				toCheck.extend(provider._getOverrideAncestors(forKey=key))
				continue
			# check if we want all matching overrides
			if not onlyFirst:
				found.append((provider, result))
				toCheck.extend(provider._getOverrideAncestors(forKey=key))
				continue
			# check if we want object or value
			if returnObj:
				found.append(provider)
			else:
				found.append(result)
			break

		if not found: # nothing found
			if not onlyFirst: # return empty list
				return found
			if default is Sentinel.FailToFind:
				raise KeyError(f"No override found for key {key} on object {self} ")
			return default

		return found[0]

