from __future__ import annotations

import traceback
import types, typing as T
import pprint
import weakref

from wplib import log

import time

from wplib.object import WpCallback
from wpm import om, cmds, WN
from wpm.core import event

class WpMayaCallback(WpCallback):
	"""test for a safer way to work with callbacks
	if you can't tell, I've had many bad experiences with zombie
	callbacks running until you restart the program,
	this absolutely will not happen here

	in this system you make an instance of this object, then attach it
	to MMessage in the same way you would a function -
	COULD go further and wrap the whole system, but seems cleaner to leave
	that divide as you would normally have it

	stayAliveFn could use more work, try adding a check here for weakRef stuff
	"""

	def remove(self):
		"""finally disconnect callback object from maya -
		this can't be undone by this object alone"""
		if self.callbackID is not None:
			om.MMessage.removeCallback(self.callbackID)
		self.callbackID = None

	def isAttached(self)->bool:
		if self.callbackID is None:
			return False
		return True


class WpMayaIdleCallback(WpMayaCallback):
	"""test for adding things to run while maya is idle -
	from the docs, a callback firing on idle events has to immediately deregister
	itself, or maya will keep calling it forever -

	this object will persist, deregister itself on firing, but RE-register
	itself based on a timer (could even use a separate maya callback timer to
	do this, to avoid using a thread

	we can't just use the timer directly because if you try to modify the scene
	while something else is happening maya hard-crashes, and the idle callback
	system is the only way I know of to tell if it's safe
	(if there's an obvious "isBusy" condition we can check for, do let me know)
	"""

	def __init__(self, fns: T.List[T.Callable[[WpCallback, ...], None]] = None, maxFPS=2.0, stayAliveFn=None):
		super().__init__(fns, maxFPS, stayAliveFn)
		self._timerCallbackID : int = None
		self.userData = None

	def _attachIdle(self):
		"""attach this specific callback to be called
		on maya's next idle event"""
		super().attach(
			om.MEventMessage.addEventCallback,

			attachPreArgs=(event.MEventMessage.idleHigh, ),
			attachPostArgs=(self.userData, )
		)

	def _detachIdle(self):
		"""remove this specific callback from idle event queue"""
		super().remove()

	def _onTimerEvent(self, dt:float, *args, **kwargs):
		"""fired by connected timer event to re-register this one"""
		#print("on cb timer event", self.callbackID)
		if self.callbackID is None:
			self._attachIdle()
			
	def __call__(self, *args, **kwargs):
		"""absolutely imperative not to block idle events
		"""
		try:
			super().__call__(*args, **kwargs)
		except Exception as e:
			log("error in idle callback, halting and deregistering "
			    "immediately", self)
			traceback.print_exc()
			self.remove()
			log("safely deregistered idle callback", self)

		finally: # always deregister this from idle callback
			self._detachIdle()


	def attach(self,
	           # mMessageAddCallbackFn,
	           idleEvent=event.MEventMessage.idle,
	           attachPreArgs:tuple=(),
	           attachPostArgs:tuple=(),
	           ):
		"""attach a secondary timer callback to keep re-attaching this
		one after we unattach after firing
		maya makes me feel alive
		"""
		#self._detachTimerCb()
		### weird jank to check this object is still alive?
		#ref = weakref.ref(self)
		# self.ownWeakRef = ref
		self._attachTimerCb()

	def _detachTimerCb(self):
		if self._timerCallbackID is not None:
			om.MTimerMessage.removeCallback(self._timerCallbackID)

	def _attachTimerCb(self):
		self._detachTimerCb()
		self._timerCallbackID = om.MTimerMessage.addTimerCallback(
			self.step, self._onTimerEvent)

	def remove(self):
		self._detachTimerCb()
		self._detachIdle()

