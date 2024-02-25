
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets


"""collapsible widget, expanding or stowing when clicked"""
class Collapsible(QtWidgets.QWidget):
	"""collapsible widget, expanding or stowing when clicked
	try to make this a mixin"""
	def __init__(self, title:str, content:QtWidgets.QWidget, parent=None):
		super().__init__(parent=parent)
		self.title = title
		self.content = content
		self.content.setParent(self)
		self.content.setVisible(False)
		self.titleBtn = QtWidgets.QPushButton(title, parent=self)
		self.titleBtn.clicked.connect(self.onTitleBtnClicked)
		self.layout = QtWidgets.QVBoxLayout()
		self.layout.addWidget(self.titleBtn)
		self.layout.addWidget(self.content)
		self.setLayout(self.layout)

	def onTitleBtnClicked(self):
		"""toggle content visibility"""
		self.content.setVisible(not self.content.isVisible())
