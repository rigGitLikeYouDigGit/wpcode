from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets

from wpui.keystate import KeyState

class DragItemWidget(QtWidgets.QWidget):
	"""small holder widget to allow dragging model items
	up and down
	also a clear way to select or deselect items?

	holding down mouse on button begins drag
	"""

	selectionClicked = QtCore.Signal()
	dragBegun = QtCore.Signal()

	def __init__(self, innerWidget:QtWidgets.QWidget, parent=None,
	             allowDrag=True, dragBeginTime=300, # milliseconds
	             ):
		super().__init__(parent)
		self.dragHandle = QtWidgets.QLabel("☰", parent=self)
		self.dragHandle.setFont(QtGui.QFont("monospace", 8))
		self.innerWidget = innerWidget
		#self.innerWidget.setParent(self)
		layout = QtWidgets.QHBoxLayout()
		layout.addWidget(self.dragHandle)
		layout.addWidget(innerWidget)
		self.setLayout(layout)
		layout.setContentsMargins(0, 0, 0, 0)
		self.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(self.dragHandle, QtCore.Qt.AlignTop)

		self.ks = KeyState()
		self._dragBeginTimer = QtCore.QTimer(parent=self)
		self._dragBeginTime = dragBeginTime
		self._dragBeginTimer.timeout.connect(
			self._onDragTimerTimeout
		)
		self._mousePressed = False
		self.setMouseTracking(True)

	def _onDragTimerTimeout(self, *args, **kwargs):
		"""fire dragBegun signal"""
		self.dragBegun.emit()
		pass

	def mousePressEvent(self, event):
		self.ks.mousePressed(event)
		if event.button() == QtCore.Qt.LeftButton:
			self._dragBeginTimer.singleShot(self._dragBeginTime)

	def mouseReleaseEvent(self, event):
		self.ks.mouseReleased(event)
		if event.button() == QtCore.Qt.LeftButton:
			# if timer is running, mouse was pressed on this widget within time limit
			shouldEmit = False
			if self._dragBeginTimer.isActive():
				shouldEmit = True
			self._dragBeginTimer.stop()
			if shouldEmit:
				self.selectionClicked.emit()

	def leaveEvent(self, event):
		"""if leave while mouse down, signal drag to start"""
		self.ks.reset()
		if self._dragBeginTimer.isActive():
			shouldEmit = True
		self._dragBeginTimer.stop()
		if shouldEmit:
			self.dragBegun.emit()



