from __future__ import annotations

import typing as T

from PySide2 import QtWidgets

from wp.ui.widget import WpWidgetBase


class ChildListWidget(QtWidgets.QWidget, WpWidgetBase):
	"""widget holding list of child widgets, instead of items"""
	def __init__(self, parent=None):
		super(ChildListWidget, self).__init__(parent=parent)
		WpWidgetBase.__init__(self)

		vl = QtWidgets.QVBoxLayout(self)
		vl.setContentsMargins(0, 0, 0, 0)
		vl.setSpacing(0)
		self.setLayout(vl)

	def addWidget(self, widget:QtWidgets.QWidget):
		"""add widget to list"""
		widget.setParent(self)
		self.layout().addWidget(widget)

	def widgets(self):
		return [self.layout().itemAt(i).widget() for i in range(self.layout().count())]

	def clear(self):
		"""clear all widgets"""
		#print("clearing list")
		while self.layout().count():
			self.layout().takeAt(0).widget().deleteWp()
		self.adjustSize()

	def widgetList(self)->T.List[QtWidgets.QWidget]:
		"""return list of widgets"""
		return [self.layout().itemAt(i).widget() for i in range(self.layout().count())]

	def sizeHint(self) -> PySide2.QtCore.QSize:
		"""expand to all list items"""
		return self.minimumSizeHint()

	def minimumSizeHint(self) -> PySide2.QtCore.QSize:
		"""expand to all list items"""
		return self.layout().minimumSize()