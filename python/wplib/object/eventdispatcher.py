
from __future__ import annotations
import typing as T

from wplib.object.signal import Signal
from wplib.sequence import flatten
from dataclasses import dataclass

"""base class defining logic for sending atomic events, handling them,
and passing them on to other objects in system

for simplicity, no accepting system - all listeners will receive all events.
if listeners want to implement more complicated systems, they can

allow subscribing to different streams of events?
"""


@dataclass
class EventBase:
	"""events can be anything, base class just used for typing direction"""
	sender : EventDispatcher = None


class EventDispatcher:
	"""base class for objects that can send events to other objects"""

	def __init__(self):
		self._eventNameSignalMap : dict[str, Signal] = {}


	def getEventSignal(self, key:str, create=False)->Signal:
		if self._eventNameSignalMap.get(key) is None:
			if create:
				self._eventNameSignalMap[key] = Signal()
		return self._eventNameSignalMap.get(key)


	def hasListeners(self):
		return self._eventNameSignalMap is not None


	def addListenerCallable(self, fn:callable, key:str):
		self.getEventSignal(key, create=True).connect(fn)


	def _nextEventDestinations(self, forEvent:EventBase, key:str)->list[EventDispatcher]:
		"""
		OVERRIDE
		return a list of objects to pass this event
		unsure if we should allow filtering here (parent deciding
		which child should receive event)
		"""
		raise NotImplementedError


	def _emitEventToListeners(self, event:EventBase, key:str):
		"""override to actually process the event on this object"""
		if self.getEventSignal(key, create=False): # event exists, emit
			self.getEventSignal(key).emit(event)


	def sendEvent(self, event:(EventBase, T.Any), key:str):
		"""user-facing entrypoint to introduce an event to the system
		should not be necessary to override this"""
		if getattr(event, "sender", None) is None:
			event.sender = self
		self._emitEventToListeners(event, key )
		for i in self._nextEventDestinations(event, key):
			i.sendEvent(event, key)


