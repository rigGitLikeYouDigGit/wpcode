from __future__ import annotations
import typing as T

import sys, os, threading

class StreamTask:
	"""This will likely be its own project, as to what a Task object should be -
	for now, keep it very simple.

	Atomic object holding contained function to be run, arguments, and return value.
	"""

	def __init__(self, func: T.Callable, argsKwargs: T.Tuple[T.Tuple, T.Dict]=((), {})):
		self.func = func
		self.argsKwargs = argsKwargs
		self.error = None
		self.result = None

	def exec(self):
		try:
			self.result = self.func(*self.argsKwargs[0], **self.argsKwargs[1])
		except Exception as e:
			self.error = e