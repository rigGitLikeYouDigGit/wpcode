
from __future__ import annotations
import typing as T

import sys
import weakref

#from .singleton import SingletonDecorator

"""manager for when multiple processes need to
override the normal excepthook

not making this a singleton, conceivably multiple could be stacked
"""

class ExceptHookManager:

	def __init__(self):
		# if a non-managed process has already overridden the excepthook
		self._prevHook : T.Callable = None
		if sys.excepthook != sys.__excepthook__:
			self._prevHook = sys.excepthook

		# register hooks with a key - in future could allow sorting
		self.hooks = weakref.WeakValueDictionary()

		sys.excepthook = self.__call__


	def registerHook(self, key:T.Any, hook:T.Callable):
		self.hooks[key] = hook

	def __call__(self, exc_type, exc_value, exc_traceback):
		"""call all hooks in order"""
		for key, hook in self.hooks.items():
			if hook is None: continue
			print("calling except hook", key)
			hook(exc_type, exc_value, exc_traceback)
		if self._prevHook:
			self._prevHook(exc_type, exc_value, exc_traceback)

	def unregisterHook(self, key:T.Any):
		del self.hooks[key]

	def clearHooks(self):
		self.hooks.clear()

	def __del__(self):
		self.clearHooks()
		# restore the previous hook
		if self._prevHook:
			sys.excepthook = self._prevHook



