from __future__ import annotations
import typing as T


from PySide2 import QtWidgets, QtCore, QtGui

from wplib.string import lowFirst

from wp.object import CleanupMixin

"""test for master base class for any widget
adds on cleanup methods, and allows hierarchy
traversal by name.

Slightly awkward since we can't actually inherit from QObject in the base
(because of Qt's MRO issues), so we have to use a mixin instead.
"""

def iterUiTreeLeavesUp(uiObject:QtCore.QObject, seen:set[int]=None)->T.Iterator[QtCore.QObject]:
	"""iterate over all widgets in a ui tree, from leaves up -
	this should traverse each object exactly once, and account for objects
	being destroyed during iteration."""

	# maximum number of iterations is starting child count -
	# objects may be deleted during iteration, so need to check against
	# seen set
	if seen is None:
		seen = set()

	for i in range(len(uiObject.children())):
		child = None
		for testChild in uiObject.children():
			if id(testChild) not in seen:
				child = testChild
				break
		if child is not None:
			yield from iterUiTreeLeavesUp(child, seen=seen)
		else:
			break
	yield uiObject
	seen.add(id(uiObject))

class WpUiBase(
		CleanupMixin,
		QtCore.QObject if T.TYPE_CHECKING else object
):
	"""Depending on what situation and what tool, I can't guarantee if
	deleteLater or cleanup will be called from top level, so each one has to call
	the other"""

	def __init__(self, cleanupOnDel=True):
		CleanupMixin.__init__(self, cleanupOnDel)
		self._qtObjectDeleted = False # set to true when qt object deleted

		# test setting up a template object name, just from class name
		self.setObjectName(lowFirst(self.__class__.__name__))

	# def _deleteWp(self):
	# 	"""internal delete method - delete any resources specifically by this widget
	# 	"""
	# 	self.cleanup()
	# 	self.

	def deleteWp(self):
		"""explicit custom delete method.
		Top-level, user facing, fire and forget - get rid of this
		widget and all its children, all its resources, forever."""
		for childObject in iterUiTreeLeavesUp(self):
			if isinstance(childObject, WpUiBase):
				childObject.cleanup()
				childObject.deleteLater()
			elif isinstance(childObject, CleanupMixin):
				childObject.cleanup()
			else:
				childObject.deleteLater()


	# def deleteLater(self) -> None:
	# 	"""set flag to indicate qt object deleted"""
	# 	self._qtObjectDeleted = True
	# 	deleteObjectTree(self)



class WpWidgetBase(
		WpUiBase,
		QtWidgets.QWidget if T.TYPE_CHECKING else object
):
	"""test systems for indexing into uis by tree path -
	tooltips of participating widgets should display name
	and path to this widget.

	We hack out the normal tooltip system to be dynamic by
	default - static setting with setToolTip() will now update an
	internal string, which is used to set the tooltip on mouseover.
	If the normal _makeToolTip() function returns anything, the static
	value will be discarded.
	"""

	def __init__(self, cleanupOnDel=True):
		super().__init__(cleanupOnDel)
		#self.setToolTip(self._tooltipPathStrings())
		self._toolTipStr = ""
		#self.setToolTip("")

		# mouse tracking used to get constant mouse events
		# these can then be used to update tooltips
		self.setMouseTracking(True)

		# mark widgets as roots, so we don't get the whole Maya ui
		# in the tooltip
		self.isWidgetRoot = False

	def mouseMoveEvent(self, event:QtGui.QMouseEvent):
		"""update tooltip on mouse move"""
		#print("base mouse move")
		self.setToolTip(self._makeToolTip() or self._toolTipStr)
		#super().mouseMoveEvent(event)

	def _makeToolTip(self)->str:
		"""EXTEND: return tooltip string for this widget"""
		return self._tooltipPathString()

	def treePath(self)->list[str]:
		"""return the path to this widget from root of ui"""
		path = []
		obj = self
		while obj is not None and (getattr(obj, "isWidgetRoot", False) == False):
			path.insert(0, obj.objectName())
			obj = obj.parent()
		return path

	def _tooltipPathString(self)->str:
		"""return tooltip data for this widget"""
		nameStr = "name: " + self.objectName() + "\n"
		pathStr = "path: " + " / ".join(self.treePath()) + "\n"
		typeStr = "type: " + self.__class__.__name__ + "\n"
		return nameStr + pathStr + typeStr


