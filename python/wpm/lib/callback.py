from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import time

from wplib.object import WpCallback
from wpm import om, cmds, WN


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
