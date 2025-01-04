from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
import sys, weakref


"""lib to manage multiple functions needing to 
run on the sys excepthook
"""

class _ExceptHookManager:

	def __init__(self):
		self.fns = []
		self.weakFns = weakref.WeakSet()

	def onSysExcHook(self, baseExceptHook, *args, **kwargs):
		log("MANAGER ON EXC HOOK", baseExceptHook, args, kwargs)
		log(self.fns)
		log(self.weakFns)
		for i in self.fns:
			i(*args, **kwargs)
		for i in self.weakFns:
			i(*args, **kwargs)
		baseExceptHook(*args, **kwargs)

	def activate(self):
		prevExceptHook = sys.excepthook
		sys.excepthook = lambda *args, **kwargs: MANAGER.onSysExcHook(
			prevExceptHook, *args, **kwargs)

MANAGER = _ExceptHookManager()




