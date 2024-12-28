from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import time

class WpCallback:
	"""base class for event-triggered callbacks in DCCs -
	surprisingly a lot can be common across DCCs,
	this began in maya, and survived largely unchanged in houdini

	TODO: maybe allow putting this in a thread?
	"""

	def __init__(self,
	             fns:T.List[T.Callable[[WpCallback, ... ], None]]=None,
	             maxFPS=2.0,
	             stayAliveFn=None,
	             #userData:dict=None
	             ):
		self.fns = fns or []
		self.maxFPS = maxFPS
		self.step = 1.0 / maxFPS
		self.t = time.time()
		self.dt = 0
		self.stayAliveFn = stayAliveFn

		self.callbackID = None # VERY IMPORTANT DO NOT FORGET THIS

		self.isPaused = False

		self.callbackMetaData = {}
		#self.userData = userData or {}

	def __call__(self, *args, **kwargs):
		t = time.time()
		dt = time.time() - self.t
		if dt < self.step:
			return
		assert self.callbackID is not None, f"ATTACH A CALLBACK ID to callback object {self} RIGHT NOW"
		if self.stayAliveFn is not None:
			# staying alive
			if not self.stayAliveFn(self, *args, **kwargs):
				# not staying alive
				self.remove()
				return
		self.t = t
		self.dt = dt
		if not self.isPaused:

			self.fire(*args, **kwargs)

	def fire(self, *cbArgs, **kwargs):
		""" top main function for cb action -
		-if multiple hookFns, trigger each one,
		-pass in dt since last activation, pass in self,

		subclass this callback object and override this,
		or just hook in functions to slots
		"""
		for i in self.fns:
			i(self, *cbArgs, **kwargs)

	def pause(self):
		self.isPaused = True
	def unpause(self):
		self.isPaused = False

	def remove(self):
		"""finally disconnect callback object from maya -
		this can't be undone by this object alone"""
		raise NotImplementedError

	def attach(self,
	           mMessageAddCallbackFn,
	           attachPreArgs:tuple=(),
	           attachPostArgs:tuple=()):
		"""
		attachPreArgs : arguments before this object in the function call
		attachPostArgs : arguments after this object in the function call (userData goes here)

		why do i make things so inelegant when I touch them

		MAYBE TOO FAR -
		a function like this lets us capture some useful id
		data about how this object is hooked up,
		PLUS the vital callback id for it"""
		if self.callbackID:
			raise RuntimeError(f"callback object {self} already attached \n ONE CALLBACK OBJECT FOR ONE CALLBACK ATTACHMENT (for now)")

		self.callbackMetaData = {
			"attachFn" : mMessageAddCallbackFn.__qualname__,
			"args" : attachPreArgs
		}
		log("attach", mMessageAddCallbackFn, attachPreArgs, self, attachPostArgs)
		callbackId = mMessageAddCallbackFn(
			*attachPreArgs, self, *attachPostArgs
		)
		self.callbackID = callbackId

	def isAttached(self)->bool:
		if self.callbackID is None:
			return False

	def __str__(self):
		return f"<{self.__class__.__name__}( {self.callbackMetaData}, {self.fns} )>"