
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets


"""collapsible widget, expanding or stowing when clicked"""
class Collapsible(QtWidgets.QWidget):
	"""collapsible widget, expanding or stowing when clicked
	try to make this a mixin"""
	expandedSignal = QtCore.Signal(bool)
	def __init__(self, title:str, w:QtWidgets.QWidget, parent=None,
	             vertical=True, expanded=True):
		super().__init__(parent=parent)
		self.title = title
		self.w = w
		self.w.setParent(self)
		self.w.setVisible(False)
		self.titleBtn = QtWidgets.QPushButton(title, parent=self)
		self.titleBtn.clicked.connect(self.onTitleBtnClicked)
		self.layout = QtWidgets.QVBoxLayout(self) if vertical else QtWidgets.QHBoxLayout(self)
		self.layout.addWidget(self.titleBtn)
		self.layout.addWidget(self.w)
		self.setLayout(self.layout)
		self.setContentsMargins(0, 0, 0, 0)
		self.layout.setContentsMargins(2, 2, 2, 2)

		self.setExpanded(expanded)

	def onTitleBtnClicked(self):
		"""toggle content visibility"""
		self.setExpanded(not self.isExpanded())

	def setExpanded(self, state=True):
		self.w.setVisible(state)
		self.expandedSignal.emit(state)

	def isExpanded(self):
		return self.w.isVisible()


class ShrinkWrapWidget(QtWidgets.QWidget):
	"""do we need ANOTHER different way of doing inheritance in widgets?
	this single behaviour obviously doesn't need a full subclass to it,
	but that's currently the only way to inject it.

	otherwise we could do a way with patching?
	so here you could do
	>>> self.label = QtWidgets.QLabel(self)
	>>> component.patch(self.label, [ShrinkWrapPatch(self.label), UIDrawingPatch(self.label)])
	?
	and internally it just assigns other functions as the patched methods on the object?
	"""

	def sizeHint(self):
		baseRect = QtCore.QRect()
		for i in self.children():
			for at in ["boundingRect", "rect"]:
				try:
					r = getattr(i, at)()
					baseRect = baseRect.united(r)
					break
				except: continue
		size = baseRect.size()
		return size
