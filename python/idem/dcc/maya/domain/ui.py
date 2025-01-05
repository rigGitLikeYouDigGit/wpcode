from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets
from idem.ui.sessionwidget import SessionWidget
from idem.dcc.maya.domain.session import MayaIdemSession

from wpm.ui import lib
class SessionWidget(SessionWidget):
	"""maya specialisation of abstract SessionWidget"""

	pass

	@classmethod
	def showWindow(cls)->SessionWidget:
		"""create floating standalone window in maya for idem
		stops idem widget disappearing behind main ui -

		could call this something like "integrateWindow()", just
		interfacing IDEM ui better with the DCC
		"""
		w = cls.getInstance()
		topWin = lib.getMainWindowWidget()
		mainWin = QtWidgets.QMainWindow(topWin)
		w.setParent(mainWin)
		mainWin.setCentralWidget(w)
		mainWin.show()
		return mainWin