
from __future__ import annotations
import typing as T


from PySide2 import QtWidgets, QtCore, QtGui

from wpm import log, cmds, om, oma, WN


"""test for persistent floating channelbox widget

- selecting plugs or nodes based on pattern
- display union or intersection of available attributes
- unreal-style "multiple values" where nodes differ 
	- some way to remap or clamp without collapsing to single value?


"""



class ChannelSlider(QtWidgets.QLineEdit):
	"""textedit mirroring Maya dragging behaviour"""

	def __init__(self, parent=None):
		super(ChannelSlider, self).__init__(parent=parent)
		self.setReadOnly(True)
		self.setDragEnabled(True)
		self.setAcceptDrops(True)
		self.setMouseTracking(True)
		self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
		self.customContextMenuRequested.connect(self.onContextMenu)
		self.setFrame(False)
		self.setFixedWidth(100)
		self.setFixedHeight(20)
		self.setStyleSheet("QLineEdit { background-color: #333; color: #fff; }")
		self.setPlaceholderText("drag to set value")
		self.setValidator(QtGui.QDoubleValidator())

		self.value = 0.0



