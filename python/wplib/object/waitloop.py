from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import time

"""loop for soft-locking competing threads - 
start a loop, optionally specify a timeout, or a min wait time

if block := waitBlock(): ? 
	do thread-unsafe stuff

if not block: # timed out
	# handle timeout ?


ok python already kind of has this (shocker) as threading.Lock and threading.RLock

"""

"""
from threading import RLock
l = RLock()
with l:
	...
"""


class WaitBlock:

	# could be interesting - raise an error in the check function
	# to flag that threaded process should stop altogether
	class ShouldAbandonError(Exception):
		pass

	def __init__(self,
	             canGoFn:T.Callable[[T.Any], bool]=lambda **kwargs : True,
	             timeout=10.0,
	             minWaitTime=None,
	             onTimeoutFn:T.Callable[[WaitBlock], None]=None,
	             onAbandonFn:T.Callable[[WaitBlock], None]=None,
	             **kwargs):
		self.canGoFn = canGoFn
		self.timeout = timeout
		self.minWaitTime = minWaitTime
		self.kwargs = kwargs
		self.onTimeoutFn = onTimeoutFn
		self.onAbandonFn = onAbandonFn
		self.timedOut = False
		self.shouldAbandon = False
		self.abandonError = None

	def __bool__(self):
		self.timedOut = False
		self.shouldAbandon = False
		self.abandonError = None
		startT = time.time()
		try:
			while not self.canGoFn(**self.kwargs):
				if self.minWaitTime is not None:
					time.sleep(self.minWaitTime)
				d = time.time() - startT
				if d > self.timeout:
					self.timedOut = True
					if self.onTimeoutFn:
						self.onTimeoutFn(self)
					return False
			return True
		except self.ShouldAbandonError as e:
			self.timedOut = False
			self.shouldAbandon = True
			self.abandonError = e
			if self.onAbandonFn:
				self.onAbandonFn(self)
			return False


