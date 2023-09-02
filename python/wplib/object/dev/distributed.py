from __future__ import annotations

import signal

from wplib.object.signal import Signal

class DistributedMirror:
	"""emulating behaviour of something like Git -
	different views of a single object,
	each able to be edited
	an edit triggers a timestamped signal -
	other views can then update themselves
	"""

	def __init__(self):
		self._listening = False
		self._emitting = False

	def setListening(self, listening:bool)->bool:
		"""set whether this mirror is listening to signals
		"""
		oldState = self._listening
		self._listening = listening
		return oldState

	def setEmitting(self, emitting:bool)->bool:
		"""set whether this mirror is emitting signals
		"""
		oldState = self._emitting
		self._emitting = emitting
		return oldState

	def signalSelfUpdated(self):
		pass