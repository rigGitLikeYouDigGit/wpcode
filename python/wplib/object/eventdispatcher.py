
from __future__ import annotations
"""base class defining logic for sending atomic events, handling them,
and passing them on to other objects in system"""
from wplib.object.signal import Signal
from wplib.sequence import flatten
from dataclasses import dataclass

@dataclass
class EventBase:
	"""events can be anything, base class just used for typing direction"""
	accepted : dict = None
	sender : EventDispatcher = None

	def __post_init__(self):
		self.accepted = self.accepted or {}


class EventDispatcher:

	def __init__(self):
		self._eventSignal : Signal = None

	@property
	def eventSignal(self)->Signal:
		if self._eventSignal is None:
			self._eventSignal = Signal()
		return self._eventSignal

	def hasListeners(self):
		return self._eventSignal is not None

	def addListenerCallable(self, fn:callable):
		self.eventSignal.connect(fn)

	def _validateEventInput(self, eventInput:EventBase, sendEventKwargs:dict)->EventBase:
		"""check event input is valid to send, optionally
		run some fixes or changes to them

		raise any error if given input is invalid
		"""
		eventInput.sender = self
		return eventInput

	def _nextEventDestinations(self, forEvent:EventBase, sendEventKwargs:dict)->list[EventDispatcher]:
		"""return a list of objects to pass this event, if it
		is not marked accepted"""
		raise NotImplementedError

	def _getEventDispatcherChain(self, forEvent:EventBase, sendEventKwargs:dict)->list[EventDispatcher]:
		"""return total list of all dispatchers to pass this event"""
		found = [self]
		nextSteps = self._nextEventDestinations(forEvent, sendEventKwargs)
		while nextSteps:
			unchecked = set(nextSteps) - set(found)
			found.extend(unchecked)
			nextSteps = set(flatten(i._nextEventDestinations(forEvent, sendEventKwargs) for i in unchecked))
		return found


	def _handleEvent(self, event:EventBase):
		"""override to actually process the event on this object"""
		if self._eventSignal:
			self.eventSignal.emit(event)
		event.accepted[self] = 1


	def _relayEvent(self, event:EventBase, targetChain=None, **sendEventKwargs):
		if targetChain is None:
			targetChain = self._getEventDispatcherChain(event, sendEventKwargs)

		# iterate through targets - no need for recursion
		while targetChain:
			handler = targetChain.pop(0)
			if event.accepted.get(handler):
				#pass
				continue
			handler._handleEvent(event)
		return event

	def sendEvent(self, event:EventBase, **sendEventKwargs):
		"""user-facing entrypoint to introduce an event to the system
		should not be necessary to override this"""
		#event = self._validateEventInput(event, sendEventKwargs)
		# setting attributes raw like this is a bit weird
		#event.sender = self
		#event.accepted = {} # set your own uses and acceptance status here?
		#todo: see if there is a better way to manage event acceptance
		return self._relayEvent(event, **sendEventKwargs)



