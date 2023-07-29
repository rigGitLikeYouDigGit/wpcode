from __future__ import annotations

from collections import defaultdict

from .cache import om

class CallbackOwner:
	"""Mixin for any objects creating Maya callbacks -
	delete callbacks when object is deleted.

	This HAS to move somewhere better

	"""

	defaultCallbackKey = "main"

	def __init__(self):
		self._callbackIds : dict[str, list[int]] = defaultdict(list)

	def addOwnedCallback(self, callback, key=defaultCallbackKey):
		"""add callback to list"""
		self._callbackIds[key].append(callback)

	def removeOwnedCallback(self, id:int, key=defaultCallbackKey):
		"""remove callback from list"""
		om.MMessage.removeCallback(id)
		self._callbackIds[key].remove(id)

	def removeAllCallbacksWithKey(self, key:str):
		"""remove all callbacks with key"""
		for i in self._callbackIds[key]:
			self.removeOwnedCallback(i, key)

	def removeAllOwnedCallbacks(self):
		"""remove all callbacks owned by this object"""
		for k in self._callbackIds:
			self.removeAllCallbacksWithKey(k)

	def __del__(self):
		self.removeAllOwnedCallbacks()

