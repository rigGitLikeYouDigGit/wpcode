from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


from PySide2 import QtCore, QtGui, QtWidgets


"""Fear of failure is a poor reason not to try - 
define an abstract base for a general view, and view elements,
to allow dragging and moving entries around, 
selecting items,
deleting, duplicating,
etc

inspired mainly by the Qt view classes, but using signals and slots
rather than a strong link to any single model?
"""

class AbstractViewItem:

	def view(self)->AbstractView:
		raise NotImplementedError

	def isSelected(self)->bool:
		return self in self.view().selectedItems()
	def isLast(self):
		return self is self.view().lastItem()

	def onSelectionStateChanged(self, newState:bool, oldState:bool):
		pass

	def onLastStateChanged(self, newState:bool, oldState:bool):
		pass


class AbstractView:
	"""

	should the selection part of this be split out
	into its own part as well?
	damn it's so crazy Qt had the exact same idea with QItemSelectionModel
	slowly just rediscovering all the reasons for this framework

	should we store item selection state on the item instead? a view item is always
	specific to this view, so
	"""

	def __init__(self):
		self._selectedItems : list[AbstractViewItem] = [] #

	def selectedItems(self)->list[AbstractViewItem]:
		return self._selectedItems

	def lastItem(self)->AbstractViewItem:
		if not self._selectedItems: return None
		return self._selectedItems[-1]

	def onBeginDrag(self, items=None):
		items = items or self.selectedItems()



