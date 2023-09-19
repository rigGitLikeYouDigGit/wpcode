

from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

"""QTableView contains private object QTableCornerButton,
no api exposed to access or override it,
reported as a very annoying issue since at least March 2007

"""

def patchPaintEvent(*args, **kwargs):
	"""patch QTableView.paintEvent to paint corner button"""
	print("patchPaintEvent", args, kwargs)
	self = args[0]
	painter = args[1]
	option = args[2]

class WPTableView(QtWidgets.QTableView):

	def getBaseCornerButton(self) ->QtWidgets.QAbstractButton:
		"""search through child qwidgets to find the corner button"""

		children = self.findChildren(QtWidgets.QWidget)
		for i in children:
			print(i, type(i), i.objectName())
		result = self.findChild(QtWidgets.QAbstractButton)
		print("result", result, type(result))
		result.paintEvent = patchPaintEvent
		return result


def onClicked(*args, **kwargs):
	print("onClicked", args, kwargs)

if __name__ == '__main__':
	import sys
	app = QtWidgets.QApplication(sys.argv)
	widget = WPTableView()
	model = QtGui.QStandardItemModel(parent=widget)
	model.setData(model.index(0, 0), "a")
	model.setData(model.index(0, 1), "b")
	model.setData(model.index(1, 0), "c")
	model.setHorizontalHeaderLabels(["A", "B"])
	model.setVerticalHeaderLabels(["1", "2"])
	widget.setModel(model)
	widget.show()

	btn = widget.getBaseCornerButton()
	# palette = btn.palette()
	# palette.setColor(QtGui.QPalette.Button, QtCore.Qt.red)
	# palette.setBrush(QtGui.QPalette.Base, QtGui.QBrush(QtCore.Qt.red))
	# palette.setBrush(QtGui.QPalette.Background, QtGui.QBrush(QtCore.Qt.red))
	# btn.setAutoFillBackground(True)
	# btn.setPalette(palette)

	btn.clicked.connect(onClicked)


	sys.exit(app.exec_())


