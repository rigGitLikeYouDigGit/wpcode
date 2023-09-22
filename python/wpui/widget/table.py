

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

class NLabel(QtWidgets.QLabel):
	pass

class WPTableView(QtWidgets.QTableView):

	def __init__(self, *args, **kwargs):
		super(WPTableView, self).__init__(*args, **kwargs)
		self._tableCornerWidget = None

	def setTableCornerWidget(self, widget:QtWidgets.QWidget):
		"""set widget to corner of table"""
		if self._tableCornerWidget:
			self._tableCornerWidget.setParent(None)
		self._tableCornerWidget = widget
		self._tableCornerWidget.setParent(self)
		self._tableCornerWidget.show()
		self._tableCornerWidget.setGeometry(self.getBaseCornerButton().geometry())

	def tableCornerWidget(self) ->QtWidgets.QWidget:
		"""return widget in corner of table"""
		return self._tableCornerWidget


	def getBaseCornerButton(self) ->QtWidgets.QAbstractButton:
		"""search through child qwidgets to find the corner button"""

		children = self.findChildren(QtWidgets.QWidget)
		for i in children:
			#print(i, type(i), i.objectName())
			pass
		result = self.findChild(QtWidgets.QAbstractButton)
		#print("result", result, type(result))
		setattr(result, "paintEvent", patchPaintEvent)
		#result.paintEvent = patchPaintEvent
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

	widget.setTableCornerWidget(NLabel("test"))

	#btn = widget.getBaseCornerButton()
	# palette = btn.palette()
	# palette.setColor(QtGui.QPalette.Button, QtCore.Qt.red)
	# palette.setBrush(QtGui.QPalette.Base, QtGui.QBrush(QtCore.Qt.red))
	# palette.setBrush(QtGui.QPalette.Background, QtGui.QBrush(QtCore.Qt.red))
	# btn.setAutoFillBackground(True)
	# btn.setPalette(palette)

	#btn.clicked.connect(onClicked)


	sys.exit(app.exec_())


