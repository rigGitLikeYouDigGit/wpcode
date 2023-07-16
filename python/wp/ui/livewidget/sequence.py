
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets

from wp.ui.livewidget.value import ValueUiElement

"""

Abstract (?) base for a ui element corresponding to a sequence - 
list, tuple, numpy array, etc.

widgetForValueFn parametre is passed to allow injection of control
for other systems - 
sequence widget only defines sequence behaviour, not how to display items

"""



class SequenceUiElement(ValueUiElement):
	"""hold, display, edit and retrieve a value.
	Inherit this class into an actual widget class when needed."""

	def append(self, value:T.Any):
		"""append value to sequence"""
		value = self.getElementValue()
		value.append(value)
		self.setElementValue(value)

	def insert(self, index:int, value:T.Any):
		"""insert value at index"""
		value = self.getElementValue()
		value.insert(index, value)
		self.setElementValue(value)

	def remove(self, value:T.Any):
		"""remove value from sequence"""
		value = self.getElementValue()
		value.remove(value)
		self.setElementValue(value)

	def setIndex(self, oldIndex:int, newIndex:int):
		"""move item at oldIndex to newIndex"""
		value = self.getElementValue()
		value.insert(newIndex, value.pop(oldIndex))
		self.setElementValue(value)


class SequenceWidget(SequenceUiElement, QtWidgets.QScrollArea):


	def __init__(self, parent=None,
	             widgetForValueFn:T.Callable[[T.Any], QtWidgets.QWidget]=None,
	             copyValue:bool=True
	             ):
		super(SequenceWidget, self).__init__(parent=parent)
		self._widgetForValueFn = widgetForValueFn
