from __future__ import annotations

from .cache import om

class CallbackOwner:
	"""Mixin for any objects creating Maya callbacks -
	delete callbacks when object is deleted.

	This HAS to move somewhere better

	"""

	def __init__(self):
		self._callbackIds = []

	def addOwnedCallback(self, callback):
		"""add callback to list"""
		self._callbackIds.append(callback)

	def removeOwnedCallback(self, id:int):
		"""remove callback from list"""
		om.MMessage.removeCallback(id)
		self._callbackIds.remove(id)

	def removeAllOwnedCallbacks(self):
		"""remove all callbacks owned by this object"""
		om.MMessage.removeCallbacks(self._callbackIds)
		self._callbackIds = []

	def __del__(self):
		self.removeAllOwnedCallbacks()


class NodeCallbackListener:
	"""may be too specific but just a test for now -
	Object listening to scene callbacks, tied to lifespan of specific node"""
