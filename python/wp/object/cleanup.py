
from __future__ import annotations
import typing as T
import types

from weakref import WeakSet, WeakValueDictionary

if T.TYPE_CHECKING:
	from tree import Signal

"""Some objects need to be explicitly cleaned up - 
ideally this would happen when all references to the object are gone,
but working in any arbitrary environment with any arbitrary methods of
integration, that's not always possible.

This is quite heavyweight, and may result in a lot of boilerplate - 
on the other hand, it will make our building blocks more robust,
and hopefully let us abstract or automate the boilerplate down the line.

Main issue here is undead signal and slot connections - 
add provision for this in the base class
"""

class CleanupMixin:
	"""call .cleanup() on this object when it is deleted -
	this is not reversible"""

	def __init__(self, cleanupOnDel:bool=True):
		"""add attribute to track if object has been cleaned up"""
		self._cleanedUp = False
		self._cleanupOnDel = cleanupOnDel
		self._weakSignalSlotMap : dict[(types.FunctionType, types.MethodType), Signal] = WeakValueDictionary()

	def connectToOwnedSlot(self, signal:Signal, slot:T.Union[types.FunctionType, types.MethodType]):
		"""run connection operator, then add reference to it in this object.
		On cleanup(), all owned signals will be disconnected"""
		signal.connect(slot)
		self._weakSignalSlotMap[(slot, signal)] = signal

	def cleanup(self):
		"""disconnect all owned signals"""
		if self._cleanedUp: # already cleaned up
			return
		#print("cleanup", dict(self._weakSignalSlotMap))
		for (slot, signal) in self._weakSignalSlotMap.keys():
			signal.disconnect(slot)
		self._cleanedUp = True

	def __del__(self):
		if self._cleanupOnDel:
			self.cleanup()
