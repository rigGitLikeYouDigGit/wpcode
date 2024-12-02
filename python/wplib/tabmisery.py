from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtWidgets, QtGui

class ShortcutEventFilter(QtCore.QObject):

	def eventFilter(self, watched, event):

		if event.type() in (QtCore.QEvent.UpdateRequest,
		                    QtCore.QEvent.Paint,
		                    QtCore.QEvent.Timer,
		                    QtCore.QEvent.DynamicPropertyChange,
		                    ):
			#return super().event(event)
			return False

		print("EVENT", event.type(), type(event))
		if event.type()==QtCore.QEvent.ShortcutOverride:
			print("eat shortcut override")
			event.accept()
			#event.ignore()
			return True
			#return False

		if isinstance(event, QtGui.QKeyEvent):
			if event.key() == QtCore.Qt.Key_Tab:
				event.accept()
				return True

		return False
		#return super().eventFilter(watched, event)


class Window(QtWidgets.QWidget):

	def __init__(self, parent=None):
		super().__init__(parent)

		self.lineA = QtWidgets.QLineEdit("lineA", self)
		self.lineB = QtWidgets.QLineEdit("lineB", self)

		# f = ShortcutEventFilter(self.lineA)
		# self.lineA.installEventFilter(f)

		vl = QtWidgets.QVBoxLayout(self)
		self.setLayout(vl)
		vl.addWidget(self.lineA)
		vl.addWidget(self.lineB)

if __name__ == '__main__':

	app = QtWidgets.QApplication()
	f = ShortcutEventFilter(app)
	app.installEventFilter(f)

	w = Window()
	w.show()

	# f = ShortcutEventFilter(w)
	# w.installEventFilter(f)

	app.exec_()




