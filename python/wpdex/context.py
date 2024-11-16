from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

"""simple objects to track a re-entrant proxy frame,
only triggering an effect when a call stack is entered and exited

I had a super complicated version of this years back, keeping this
simple since it's used in conjunction with the proxy wrapper
"""


class ReentrantContext:
	def __init__(self,
	             onTopEnterFn=None,
	             onTopExitFn=None,
	             onAnyEnterFn=None,
	             onAnyExitFn=None):
		self.depth = 0
		self.onTopEnterFn = onTopEnterFn
		self.onTopExitFn = onTopExitFn
		self.onAnyEnterFn = onAnyEnterFn
		self.onAnyExitFn = onAnyExitFn

	def __enter__(self, *args, **kwargs):
		if self.depth == 0:
			if self.onTopEnterFn is not None:
				self.onTopEnterFn(*args, **kwargs)
		self.depth += 1

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.depth -= 1
		#if self.on
		if self.depth == 0:
			if self.onTopExitFn is not None:
				self.onTopExitFn(exc_type, exc_val, exc_tb)

	def onTopEnter(self):
		pass

	def onTopExit(self, exc_type=None, exc_val=None, exc_tb=None):
		if exc_type:
			raise exc_val
		pass



