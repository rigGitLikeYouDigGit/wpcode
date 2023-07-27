from __future__ import annotations
import typing as T
from weakref import WeakSet

from PySide2 import QtWidgets, QtCore, QtGui

from wp.object import StatusObject, ErrorReport, CleanupMixin

from wp.ui.widget import StringWidget


from wpm import cmds, om, oma, WN, createWN, getSceneGlobals
from wpm.lib.validation import nodeNameRuleSet

"""widget representing a single maya node in scene, 
flagging error if node cannot be found, or is deleted"""

class NodeTracker(StringWidget):
	"""single click on label selects node in scene"""

	def __init__(self, parent=None):
		StringWidget.__init__(self, parent)

		self._node : WN = None

		self.setValidationRuleSet(nodeNameRuleSet)

	def node(self):
		return self._node


	def onNodeRenamed(self, *args, **kwargs):
		if self._cleanedUp:
			return # already deleted
			raise RuntimeError(f"NodeTracker already deleted")
		if self._node is None:
			return
		if self.isEditing():
			return
		self.setValue(self.node().name())

	def setNode(self, node:WN):
		"""connect to node callback, listening for name changes"""
		#self._nodeSet.add(node)
		# if node is None:
		#
		self._node = node
		self.setValue(node.name())
		self.connectToOwnedSlot(self.node().getNodeNameChangedSignal(),
		                        self.onNodeRenamed)
		#self.node().getNodeNameChangedSignal().connect(self.onNodeRenamed)


	def _onValueChanged(self, text:str):
		if text != self.node().name():
			self.node().setName(text)

	def deleteLater(self) -> None:
		self.cleanup()
		self._node = None
		super(NodeTracker, self).deleteLater()



