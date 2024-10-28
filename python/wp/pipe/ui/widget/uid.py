
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wp.ui.widget import WpWidgetBase, BorderFrame

"""widget for displaying uid value in compact way, allowing for copying.
May also add one to allow direct entry, but that needs validation, and
might overlap with string widget.

Maybe on mouse over can bring up asset info window on given uid
"""

class UidWidget(QtWidgets.QLabel,
                BorderFrame,
                ):
	"""label directly displaying truncated uid

	TODO: REWORK ALL OF THIS with proper reactive base
		probably way more simple
	"""
	def __init__(self, uid:str="", parent=None):
		QtWidgets.QLabel.__init__(self, parent=parent)
		self._uid = ""
		if uid:
			self.setUid(uid)

		#self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
		# size policy
		pol = QtWidgets.QSizePolicy(
			QtWidgets.QSizePolicy.Fixed,
			QtWidgets.QSizePolicy.Fixed,
		)
		self.setSizePolicy(pol)


	def paintEvent(self, arg__1:PySide2.QtGui.QPaintEvent) -> None:
		super(UidWidget, self).paintEvent(arg__1)
		BorderFrame.paintEvent(self, arg__1)

	def setUid(self, uid:str):
		self._uid = uid
		self.setText(" " + uid[:4] + "...")
		self.setToolTip(uid)

	def uid(self)->str:
		return self._uid

	def copy(self):
		"""copy uid to clipboard"""
		clipboard = QtWidgets.QApplication.clipboard()
		clipboard.setText(self._uid)

	def contextMenuEvent(self, ev:PySide2.QtGui.QContextMenuEvent) -> None:
		menu = QtWidgets.QMenu(self)
		menu.addAction("Copy", self.copy)
		menu.exec_(ev.globalPos())