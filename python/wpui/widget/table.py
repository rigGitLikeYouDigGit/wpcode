

from __future__ import annotations
import typing as T
import random, time, math, threading

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
	"""easier interaction with top left corner of table"""
	_tableCornerWidget: QtWidgets.QWidget

	def setTableCornerWidget(self, widget:QtWidgets.QWidget):
		"""set widget to corner of table"""
		if self.tableCornerWidget():
			self.tableCornerWidget().setParent(None)
		self._tableCornerWidget = widget
		self._tableCornerWidget.setParent(self)
		self._tableCornerWidget.show()
		self._tableCornerWidget.setGeometry(self.getBaseCornerButton().geometry())

	def tableCornerWidget(self) ->QtWidgets.QWidget:
		"""return widget in corner of table"""
		return getattr(self, "_tableCornerWidget", None)


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

	# def resizeEvent(self, event):
	# 	self.resizeRowsToContents()
	# 	#self.updateGeometries()
	#
	# 	super().resizeEvent(event)

"""system for embedding widgets within table view
and resizing correctly.

factors out the sizing from the main super item system

for real though is there ever a good reason NOT to use a tree view?

"""

class WPTreeView(QtWidgets.QTreeView):
	"""
	- easier interaction with top left corner of table
	- proper support for resizing items and index widgets
		for each row / column * x / y,
		required size is maximum of each index size data, and any index widgets on that row/column
	"""
	_tableCornerWidget: QtWidgets.QWidget

	def setTableCornerWidget(self, widget:QtWidgets.QWidget):
		"""set widget to corner of table"""
		if self.tableCornerWidget():
			self.tableCornerWidget().setParent(None)
		self._tableCornerWidget = widget
		self._tableCornerWidget.setParent(self)
		self._tableCornerWidget.show()
		self._tableCornerWidget.setGeometry(self.getBaseCornerButton().geometry())

	def tableCornerWidget(self) ->QtWidgets.QWidget:
		"""return widget in corner of table"""
		return getattr(self, "_tableCornerWidget", None)

	def syncLayout(self, execute:bool=False):
		"""sync layout of the table"""
		self.scheduleDelayedItemsLayout()
		if execute:
			self.executeDelayedItemsLayout()

	# these helpfully-named methods never get called
	# def sizeHintForIndex(self, index):
	# 	"""return the size hint for the index"""
	# 	print("size hint for index", index.row(), index.column())
	# 	#return QtCore.QSize(100, 100)
	# 	#return super().sizeHintForIndex(index)
	# 	size = 100.0 * abs(math.sin(time.time() * 0.5))
	# 	#print("size hint role", index.row(), index.column(), size)
	# 	return QtCore.QSize(100, size)
	#
	# def indexRowSizeHint(self, index):
	# 	"""return the size hint for the row"""
	# 	print("index row size hint", index)
	# 	return 100.0 * abs(math.sin(time.time() * 0.5))




def getSinSize(t:float)->float:
	"""update the size of the widget"""
	return 100.0 * abs(math.sin(t * 0.5)) + 10

class ResizableWidget(QtWidgets.QWidget):
	"""test if resizing by item data works"""

	def __init__(self, *args, **kwargs):
		"""init"""
		super().__init__(*args, **kwargs)
		#self._size = (100, 100)
		self.setAutoFillBackground(True)

	# def setSizeHint(self, size:tuple):
	# 	"""set the size hint"""
	# 	self._size = size
	# 	self.updateGeometry()

	def sizeHint(self):
		"""return the size hint"""
		#print("size hint")
		return QtCore.QSize(100, int(getSinSize(time.time())))


_base = QtCore.QAbstractTableModel
if not T.TYPE_CHECKING:
	_base = object

class ResizableModel(QtGui.QStandardItemModel
                     ):

	"""test if resizing by item data works"""

	def data(self, index, role=...):

		if role == QtCore.Qt.SizeHintRole:

			size = 100.0 * abs(math.sin(time.time() * 0.5))
			#print("size hint role", index.row(), index.column(), size)
			size = max(size, 10)
			return QtCore.QSize(100, size)
		return super().data(index, role)







def onClicked(*args, **kwargs):
	print("onClicked", args, kwargs)

if __name__ == '__main__':
	import sys
	import qt_material
	app = QtWidgets.QApplication(sys.argv)
	qt_material.apply_stylesheet(app, theme='dark_blue.xml')

	#widget = WPTableView()
	#widget = QtWidgets.QTreeView()
	widget = WPTreeView()
	#widget.setAnimated(True)
	#model = ResizableModel(parent=widget)
	model = QtGui.QStandardItemModel(parent=widget)

	def _syncSize(*args, **kwargs):
		#print("sync size", args, kwargs)
		while True:
			#print("resize rows to contents")
			#widget.resizeRowsToContents()
			#widget.resizeColumnToContents(0)
			#widget.setDirtyRegion(widget.viewport().rect())
			#widget.viewport().update()
			#widget.updateGeometries()
			#widget.header().updateGeometries()
			#widget.updateGeometry()
			#widget.update()
			#widget.adjustSize()
			#widget.viewport().update()
			#widget.columnResized(0, 0, 0)
			#widget.resizeColumnToContents(0)
			#widget.update(model.index(0, 0))



			#widget.dataChanged(model.index(0, 0), model.index(0, 0))
			# WORKS but blocks data entry


			#widget.dataChanged(model.index(-1, -1), model.index(-1, -1))

			#widget.updateGeometries()

			####### THIS IS IT
			#widget.scheduleDelayedItemsLayout()
			#widget.executeDelayedItemsLayout()
			#######

			#widget.viewport().update()
			#widget.update(model.index(1, 0))

			# widget.drawTree(
			# 	widget.sharedPainter(),
			# 	widget.visualRect(widget.model().index(0, 0)),
			# )

			# widget.repaint() # crashes
			# widget.viewport().repaint() # crashes

			widget.syncLayout(execute=True)

			time.sleep(0.041)

	widget.setModel(model)
	# model.setData(model.index(0, 0), "a")
	# model.setData(model.index(0, 1), "b")
	# model.setData(model.index(1, 0), "c")
	model.setHorizontalHeaderLabels(["A", "B"])
	model.appendRow([
		QtGui.QStandardItem("a"),
		QtGui.QStandardItem("b"),
	])
	model.appendRow([
		QtGui.QStandardItem("a"),
		QtGui.QStandardItem("b"),
	])
	# add nested row
	item = model.itemFromIndex(model.index(1, 0))
	item.appendRow([
		QtGui.QStandardItem("c"),
		QtGui.QStandardItem("d"),
	])

	# set widget
	nestedItem = model.itemFromIndex(model.index(1, 0)).child(0, 0)
	embedWidget = ResizableWidget()
	widget.setIndexWidget(nestedItem.index(), embedWidget)

	model.appendRow([
		QtGui.QStandardItem("a"),
		QtGui.QStandardItem("b"),
	])
	widget.show()

	# thread to resize
	t = threading.Thread(target=_syncSize, daemon=True).start()
	print("thread started", t)




	sys.exit(app.exec_())


