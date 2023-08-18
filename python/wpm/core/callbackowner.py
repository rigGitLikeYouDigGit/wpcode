from __future__ import annotations

from collections import defaultdict

from .cache import om

class CallbackOwner:
	"""Mixin for any objects creating Maya callbacks -
	delete callbacks when object is deleted.

	This HAS to move somewhere better

	"""

	DEFAULT_CB_KEY = "main"

	def __init__(self):
		self._callbackIds : dict[str, list[int]] = defaultdict(list)

	def addOwnedCallback(self, callback, key=DEFAULT_CB_KEY):
		"""add callback to list"""
		self._callbackIds[key].append(callback)

	def removeOwnedCallback(self, id:int, key=DEFAULT_CB_KEY):
		"""remove callback from list"""
		om.MMessage.removeCallback(id)
		self._callbackIds[key].remove(id)

	def removeAllCallbacksWithKey(self, key:str):
		"""remove all callbacks with key"""
		for i in tuple(self._callbackIds[key]):
			self.removeOwnedCallback(i, key)

	def removeAllOwnedCallbacks(self):
		"""remove all callbacks owned by this object"""
		#print("remove all owned callbacks", self._callbackIds)
		for k in self._callbackIds:
			self.removeAllCallbacksWithKey(k)

		#print("done")
		#print(self._callbackIds)

	def __del__(self):
		self.removeAllOwnedCallbacks()

