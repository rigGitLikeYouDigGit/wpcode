
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel
from wpui.superitem.model import SuperModel

"""item -> model -> item"""




class SuperDelegate(QtWidgets.QStyledItemDelegate):
	"""delegate for superitem"""

	# def __init__(self, parent=None):
	# 	super(SuperDelegate, self).__init__(parent)

	def sizeHint(self, option:PySide2.QtWidgets.QStyleOptionViewItem, index:PySide2.QtCore.QModelIndex) -> PySide2.QtCore.QSize:
		"""return size hint for index - if complex, delegate to nested widget
		"""
		#print("size hint", index)
		if self.parent().indexWidget(index):
			item : SuperItem = self.parent().model().itemFromIndex(index)
			#print("size hint", item, item.childWidget.sizeHint())
			if item.childWidget:
				return item.childWidget.sizeHint()

		return super(SuperDelegate, self).sizeHint(option, index)

	# def paint(self, painter:PySide2.QtGui.QPainter, option:PySide2.QtWidgets.QStyleOptionViewItem, index:PySide2.QtCore.QModelIndex) -> None:
	# 	"""paint the item - if complex, delegate to nested widget
	# 	"""
	# 	#print("paint", index)
	# 	# if self.parent().indexWidget(index):
	# 	# 	#self.sizeHintChanged.emit(index)
	# 	# 	return self.parent().indexWidget(index).paint(painter, option, index)
	# 	return super(SuperDelegate, self).paint(painter, option, index)

