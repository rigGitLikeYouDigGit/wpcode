from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
import numpy as np

class IndexScope:
	indices : np.ndarray
	def __init__(self, name:str):
		"""maintain dict of scope maps as needed -

		toScopeIndexMap[ other scope name][ this scope index] = other scope index
		"""
		self.name = name
		self.hashIndexMap = {}


class IndexScopeGroup:
	"""map indices between different nodes, for any type of index -
	this is a generalisation of skin influence mapping"""

	def __init__(self):
		self.hashGlobalIndexMap = {}
		self.toScopeIndexMap: dict[str, np.ndarray] = {}
		self.fromScopeIndexMap: dict[str, np.ndarray] = {}

	@classmethod
	def fromScopes(cls, scopes:tuple[IndexScope]):
		"""derive global index map from given scopes -
		this is the main function of this class, and should be used to
		generate the global map for any type of index mapping"""
		obj = cls()
		for i in scopes:
			obj.syncScope(i)

	def syncScope(self, scope:IndexScope):
		"""add new scope to group, updating global map and all existing scope maps.
		If a hash is no longer used, it is not removed - this may lead to
		dead indices. likely fine
		"""
		self.toScopeIndexMap[scope.name] = np.full(
			len(scope.hashIndexMap), -1, dtype=int)
		self.fromScopeIndexMap[scope.name] = np.zeros(len(scope.hashIndexMap), dtype=int)
		for h, localIndex in scope.hashIndexMap.items():
			if h not in self.hashGlobalIndexMap:
				self.hashGlobalIndexMap[h] = len(self.hashGlobalIndexMap)
			globalIndex = self.hashGlobalIndexMap[h]
			self.toScopeIndexMap[scope.name][globalIndex] = localIndex
			self.fromScopeIndexMap[scope.name][localIndex] = globalIndex
