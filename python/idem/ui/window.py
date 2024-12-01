

from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import Sentinel, log
from wpdex import *
from .instance import IdemWidget, IdemSession
"""
each package can define an "onStartup" function,
fully immersed in dcc domain code, and idem calls it
once dcc is running


"""

checkEventTypes = (
			QtGui.QActionEvent,
			QtGui.QKeyEvent,
			QtGui.QShortcutEvent,
			#QtGui.QInputEvent, # superclass for mouse and keyboard events
			QtGui.QInputMethodEvent,
			QtGui.QInputMethodQueryEvent,
			QtGui.QFocusEvent

		)
class ShortcutFilter(QtCore.QObject):

	def eventFilter(self, watched, event):
		if not isinstance(event, checkEventTypes):
			return False

		# log("APP EVENT", type(event), event.type())
		# if isinstance(event, QtGui.QInputMethodQueryEvent):
		# 	log("input method", event.queries())
		# 	# for i in event.queries():
		# 	# 	print(i)

		if event.type()==QtCore.QEvent.Type.ShortcutOverride:
			log("SHORTCUT OVERRIDE", watched, event.type(), event.key())
			event.accept()
			return True

		if isinstance(event, QtGui.QShortcutEvent):
			log("SHORTCUT", watched, event.key())
			event.accept()
			return True
		#super().eventFilter(watched, event)
		return False

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

		shortcutFilter = ShortcutFilter(app)
		app.installEventFilter(shortcutFilter)


		"""track down any shortcuts set up by default and REMOVE THEM"""
		### nothing on the app
		# shortcuts = app.findChildren(QtWidgets.QShortcut)
		# log("found shortcuts", shortcuts)
		# for i in shortcuts:
		# 	print(i)
		# raise

		w = cls()
		w.show()

		# dummyShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Tab),
		#                                     w)

		## nothing on the WINDOW either
		shortcuts = (app.findChildren(QtWidgets.QShortcut,)# QtCore.Qt.FindChildrenRecursively)
		             + app.findChildren(QtWidgets.QAction,)# QtCore.Qt.FindChildrenRecursively)
		             + app.findChildren(QtWidgets.QActionGroup,)# QtCore.Qt.FindChildrenRecursively)
		             )

		shortcutTypes = (QtWidgets.QShortcut,
		                 QtWidgets.QAction,
		                 QtWidgets.QActionGroup)

		toSearch = [app] + app.topLevelWidgets() + app.topLevelWindows()
		while toSearch:
			t = toSearch.pop(-1)
			if isinstance(t, shortcutTypes):
				log("FOUND SHORTCUT", type(t), t)
			if isinstance(t, QtWidgets.QWidget):
				# t.setTabOrder(t, t)
				t.installEventFilter(ShortcutFilter(t))
			toSearch.extend(t.children())


		log("found shortcuts", shortcuts)
		for i in shortcuts:
			print(i)
		#raise



		node = w.w.graphW.scene.graph().createNode("MayaSessionNode")
		# test for moving - layout needs to be more fluid
		w.w.graphW.scene.delegatesForObj(node).pop().setPos(
			w.w.graphW.view.sceneRect().center() / 3.0
		)
		#
		# def _onFocusChanged(*args):
		# 	log("onFocusChanged", args)
		# app.focusChanged.connect(_onFocusChanged)

		app.exec_()

