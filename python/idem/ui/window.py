

from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import Sentinel, log
from wpdex import *
from .instance import IdemWidget, IdemSession

from wpui.lib import AppTabFilter
"""
each package can define an "onStartup" function,
fully immersed in dcc domain code, and idem calls it
once dcc is running


"""


class Window(QtWidgets.QWidget):
	"""
	instead of multiple sessions in tabs,
	maybe it's less complicated to
	just open multiple idem windows
	"""

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setObjectName("idem")
		self.setWindowTitle("idem")

		# # list of active sessions
		self.sessions = [IdemSession.create(name="idem")]
		self.w = IdemWidget(self.sessions[0], parent=self)

		l = QtWidgets.QVBoxLayout(self)
		self.setLayout(l)
		l.addWidget(self.w)
		self.setContentsMargins(0, 0, 0, 0)


	@classmethod
	def launch(cls):
		"""overall method to start a qt app from scratch
		"""
		app = QtWidgets.QApplication()
		from wpui.theme.dark import applyStyle
		#app.setStyleSheet(STYLE_SHEET)
		applyStyle(app)

		shortcutFilter = AppTabFilter(app)
		app.installEventFilter(shortcutFilter)

		w = cls()
		w.show()

		node = w.w.graphW.scene.graph().createNode("MayaSessionNode")
		# test for moving - layout needs to be more fluid
		w.w.graphW.scene.delegatesForObj(node).pop().setPos(
			w.w.graphW.view.sceneRect().center() / 3.0
		)

		app.exec_()

