from __future__ import annotations
import types, typing as T
import pprint

import hou
from wplib import log

from wplib.object import WpCallback


class WpHoudiniCallback(WpCallback):

	node : hou.Node
	eventTypes : tuple[hou.nodeEventType]
	def attach(self,
	           mMessageAddCallbackFn,
	           attachPreArgs:tuple=(),
	           attachPostArgs:tuple=()):
		"""hook up event by the given function"""
		super().attach(mMessageAddCallbackFn,
		               attachPreArgs,
		               attachPostArgs)

		self.callbackID = self # houdini allows you to remove a cb object directly
		self.node = mMessageAddCallbackFn.__self__
		self.eventTypes = attachPreArgs

	def remove(self):
		self.node.removeEventCallback(self.eventTypes, self)



