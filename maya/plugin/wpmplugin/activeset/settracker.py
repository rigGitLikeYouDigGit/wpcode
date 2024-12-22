
from __future__ import annotations
import typing as T

import os, sys
from pathlib import Path

from wpm import om
from wpm.lib.tracker import NodeLifespanBound
from wplib.expression import Expression

"""lifespan watcher to keep set updated,
listening to scene messages and checking set expression results
whenever:
- a node is added to the scene
- a node is removed from the scene
- a node is renamed
- a node is reparented
- main set node's connections change
- another object set's membership changes
"""

class ActiveSetLifespanTracker(NodeLifespanBound):
	"""callback listener for lifespan of active set node"""

	SET_KEY = "activeSet"

	def __init__(self, node):
		super(ActiveSetLifespanTracker, self).__init__(node)
		self._setExpression : Expression = Expression(name="activeSetExpression")

	def updateSet(self, *args, **kwargs):
		"""update set membership"""
		if self.isPaused():
			#print("updateSet paused, skipping")
			return

		# get set expression string
		mfn = om.MFnDependencyNode(self.node())
		setExpressionStr = mfn.findPlug("setExpression").asString()

		# check if it's changed - if so, recompile expression
		if self._setExpression.getText() != setExpressionStr:
			self._setExpression.setSourceText(setExpressionStr)




		pass

	def filterSetPlugConnectionFromCallback(
			self, plug:om.MPlug, otherPlug:om.MPlug,
			*args, **kwargs):
		"""filter callback for set plug connections -
		only process if the other plug is a set plug,
		or if it directly involves activeSet
		"""
		#print("filterSetPlugConnectionFromCallback", plug, otherPlug, args, kwargs)
		if "SetMembers" in otherPlug.name() or otherPlug.node() == self.node():
			self.updateSet(plug, otherPlug, *args, **kwargs)

	def _connectActiveSetCallbacks(self, node):
		"""connect callbacks to active set node -
		I think it should be enough to only listen for dag nodes here"""

		self.addOwnedCallback(
			om.MDGMessage.addNodeAddedCallback(
				self.updateSet, "dagNode"),
			key=self.SET_KEY
		)

		self.addOwnedCallback(
			om.MDGMessage.addNodeRemovedCallback(
				self.updateSet, "dagNode"),
			key=self.SET_KEY
		)

		# passing empty object listens to all node name changes
		self.addOwnedCallback(
			om.MNodeMessage.addNameChangedCallback(
				om.MObject(), self.updateSet),
			key=self.SET_KEY
		)

		self.addOwnedCallback(
			om.MNodeMessage.addAttributeChangedCallback(
				self.node(), self.updateSet),
			key=self.SET_KEY
			)

		# listen to connection changes, since you can't specifically filter for sets
		self.addOwnedCallback(
			om.MDGMessage.addConnectionCallback(
				self.filterSetPlugConnectionFromCallback),
			key=self.SET_KEY
		)

	def onAttach(self):
		"""called when attached to node"""
		self._connectActiveSetCallbacks(self.node())



