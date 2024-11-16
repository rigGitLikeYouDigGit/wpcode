
from __future__ import annotations

import traceback
import typing as T

from wplib.log import log
from wplib.object.signal import Signal
from wplib.sequence import flatten
from dataclasses import dataclass

"""base class defining logic for sending atomic events, handling them,
and passing them on to other objects in system

for simplicity, no accepting system - all listeners will receive all events.
if listeners want to implement more complicated systems, they can

allow subscribing to different streams of events?
"""


class EventDispatcher:
	"""base class for objects that can send events to other objects"""

	def __init__(self):
		self._eventNameSignalMap : dict[str, Signal] = {}

	def __hash__(self):
		return id(self)

	@classmethod
	def _newEventDict(cls, sender, **kwargs):
		"""just return the literal kwargs here?"""
		return {"sender" : sender, **kwargs}


	def getEventSignal(self, key:str="main", create=True)->Signal:
		"""get the signal that will emit an event, for a given key"""
		if self._eventNameSignalMap.get(key) is None:
			if create:
				self._eventNameSignalMap[key] = Signal(name="event_" + key)
		return self._eventNameSignalMap.get(key)


	def hasListeners(self):
		return self._eventNameSignalMap is not None


	def addListenerCallable(self, fn:callable, key:str):
		self.getEventSignal(key, create=True).connect(fn)


	def _nextEventDestinations(self, forEvent:dict, key:str)->list[EventDispatcher]:
		"""
		OVERRIDE
		return a list of objects to pass this event
		unsure if we should allow filtering here (parent deciding
		which child should receive event)
		"""
		raise NotImplementedError

	def _allEventDestinations(self, forEvent:dict, key:str)->list[EventDispatcher]:
		"""use to prevent recursive sendEvent calls -
		look at all destinations first, return flat list"""
		sources = [self]
		toSend = [self]

		while sources:
			source = sources.pop(0)
			destinations = source._nextEventDestinations(forEvent, key)
			toSend.extend(destinations)
			sources.extend(destinations)
		return toSend



	def _handleEvent(self, event:dict, key:str= "main"):
		"""override to actually process the event on this object -
		if a signal is found for that key, emit the event
		to that signal's listeners
		"""
		#log("handling event", self)
		#log(event, key, self.getEventSignal(key, create=False))
		if self.getEventSignal(key, create=False): # event exists, emit
			self.getEventSignal(key).emit(event)


	def sendEvent(self, event:dict, key:str="main"):
		"""user-facing entrypoint to introduce an event to the system
		should not be necessary to override this"""
		#return
		if event.get("sender") is None:
			event["sender"] = self

		try:
			for i in self._allEventDestinations(event, key):
				i._handleEvent(event, key)
		except Exception as e:
			print("error in event handling", e)
			traceback.print_exc()
		# for i in self._nextEventDestinations(event, key):
		# 	i.sendEvent(event, key)


