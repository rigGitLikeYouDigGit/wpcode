from __future__ import annotations
import typing as T


from PySide2 import QtCore, QtGui, QtWidgets

class IndexItem(QtWidgets.QTableWidgetItem):

	def __init__(self, parent=None):
		super(IndexItem, self).__init__(parent)
		self.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled)


class IndexList(QtWidgets.QTableWidget):

	def __init__(self, parent=None):
		super(IndexList, self).__init__(parent)
		self.setItemPrototype(IndexItem())
		self.setVerticalHeaderLabels(["Index", "Name"])
		self.setDragEnabled(True)
		self.setAcceptDrops(True)
		self.setDropIndicatorShown(True)
		self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)


if __name__ == '__main__':
	import sys
	app = QtWidgets.QApplication(sys.argv)
	widget = IndexList()

	widget.insertRow(0)
	widget.insertRow(1)
	widget.setItem(0, 0, IndexItem("a"))

	# widget.addItem("a")
	# widget.addItem("b")
	# widget.addItem("c")
	# widget.addItem("d")
	# widget.addItem("e")
	widget.show()
	sys.exit(app.exec_())

