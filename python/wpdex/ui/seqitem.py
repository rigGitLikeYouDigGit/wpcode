
from __future__ import annotations

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log

from wpdex import WpDex, SeqDex, WpDexProxy
from wpdex.ui.atomic import AtomicWidget
from wpdex.ui.base import WpDexView, DexViewExpandButton


class DragItemWidget(QtWidgets.QWidget):
	"""small holder widget to allow dragging model items
	up and down"""
	def __init__(self, innerWidget:QtWidgets.QWidget, parent=None):
		super().__init__(parent)
		self.dragHandle = QtWidgets.QLabel("☰", parent=self)
		self.dragHandle.setFont(QtGui.QFont("monospace", 8))
		self.innerWidget = innerWidget
		#self.innerWidget.setParent(self)
		layout = QtWidgets.QHBoxLayout()
		layout.addWidget(self.dragHandle)
		layout.addWidget(innerWidget)
		self.setLayout(layout)
		layout.setContentsMargins(0, 0, 0, 0)
		self.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(self.dragHandle, QtCore.Qt.AlignTop)



def sketch():

	d = myObject.ref("baseData")
	w = Widget(d, parent=None)
	# THAT'S IT. that's all.
	# maybe later add some path override stuff
	# to say if file paths or strings should be taken, etc
	#

def childView(self, forDex):

	childW = AtomicWidget.adaptorForObj(forDex)(value=forDex, parent=self)
	childW.valueChanged.connect(self.sync)

def onChildValueChanged(key:keyT):
	"""regenerate only the widget at this path"""

def onValueChanged():
	"""remove / regenerate all child widgets"""


class _SeqDexModel(QtGui.QStandardItemModel):
	pass


#class SeqDexView(AtomicWidget, QtWidgets.QTreeView):
class SeqDexView(QtWidgets.QTreeView,# AtomicWidget
                 WpDexView
                 ):
	"""view for a list

	drag behaviour -
	drop directly on top of an item: override or swap?
	drop between items: insert

	or...
	ignore drag/drop for now, emulate something like a cursor in vim

	click an item to select it
	press enter to edit it
	ctl-v overwrites a selected item
	press alt to switch to cursor between items?

	work with selection spans / cursor spans - each has a start and end index


	"""
	forTypes = (SeqDex,)

	def __init__(self, value, parent=None):
		QtWidgets.QTreeView.__init__(self, parent)
		WpDexView.__init__(self, value)
		#log("seq init")
		a = 1

		self.header().setDefaultSectionSize(2)
		#self.header().setMinimumSectionSize(-1) # sets to font metrics, still buffer around it
		self.header().setMinimumSectionSize(15)
		self.header().setSectionResizeMode(
			self.header().ResizeToContents
		)
		self.setColumnWidth(0, 2)

		self.setIndentation(0)

		self.setSizeAdjustPolicy(
			QtWidgets.QAbstractItemView.AdjustToContents)
		self.setHeaderHidden(True)

		self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
		self.setHorizontalScrollMode(self.ScrollMode.ScrollPerPixel)

		self.postInit()
		self.setContentsMargins(0, 0, 0, 0)
		self.setViewportMargins(0, 0, 0, 0)

		self.setUniformRowHeights(False)


	def modelCls(self):
		return _SeqDexModel
	def buildChildWidgets(self):
		"""populate childWidgets map with widgets
		for all dex children
		acts as general sync method """
		dex = self.dex()
		#log("buildChildWidgets")
		self.setModel(self.modelCls()(parent=self))

		# clear any child widgets
		ties = tuple(self._childAtomics.items())
		self._childAtomics.clear()
		for k, w in ties:
			w.setParent(None)
			w.deleteLater()



		for key, child in self.dex().branchMap().items():
			childWidgetType = AtomicWidget.adaptorForObject(child)
			assert childWidgetType, f"no widget type found for dex {child}, parent {self.dex()}, self"
			childWidget = childWidgetType(value=child,
			                              parent=self,
			                              #parent=None,
			                              )
			#log("childWidget", childWidget, vars=0)
			self.model().appendRow(
				[QtGui.QStandardItem(), QtGui.QStandardItem(str(key))]
			)
			newIndex = self._modelIndexForKey(key)
			#log("newIndex", newIndex, child)
			#self.model().setItemData(newIndex, )
			#self._childAtomics[key] = childWidget
			self.setIndexWidget(self._modelIndexForKey(key),
			                    childWidget)
			self._setChildAtomicWidget(key, childWidget)


		# rootItem : QtGui.QStandardItem = self.model().itemFromIndex(self.model().index(0, 0))
		# rootItem.setText("[")
		topLeftIndex = self.model().index(0, 0)
		openChar = "[" if isinstance(self.dex().obj, list) else "("
		label = DexViewExpandButton(openChar,
		                            dex=self.dex(), parent=self)
		label.expanded.connect(self._setValuesVisible)
		#label.clicked.connect(self._toggleValuesVisible)

		self.setIndexWidget(topLeftIndex, label)
		self.syncLayout()

	def _setValuesVisible(self, state=True):
		self.setColumnHidden(1,
		                     not state
		                     )
		self.resizeColumnToContents(0)
		self.resizeColumnToContents(1)
		self.update()
		self.updateGeometry()
		self.syncLayout()
		self.parent().updateGeometry()
		if isinstance(self.parent(), WpDexView):
			self.parent().syncLayout()
			self.parent().updateGeometries()
			self.parent().syncLayout()




# class ListWpItemWidget(WpItemWidget):
# 	"""widget for a list"""
# 	forTypes = SEQ_TYPES
#
# 	superView : ListWpDexView
#
# 	def __init__(self, superItem:ListSuperItem, parent=None,
# 	             expanded=True):
# 		super().__init__( superItem, parent=parent)
#
# 		# initialise view
# 		self.superViewType = superItem._getComponentTypeForObject(
# 			self.superItem.wpPyObj, "view")
# 		self.superView = self.superViewType(
# 			self.superItem, parent=self
# 		)
# 		self.superView.setModel(self.superItem.wpChildModel)
# 		self.superView.buildIndexWidgets()
# 		# layout = QtWidgets.QVBoxLayout(self)
# 		# layout.addWidget(self.superView)
# 		# self.setLayout(layout)
#
# 		self.makeLayout()
#
# 		log("list w init", self.typeLabel)
#
# 		# set expanded state
# 		#self.setExpanded(expanded)
#
# 		#self.resize(self.sizeHint())
#
#
# 		#self.resize(300, 300)
# 		#self.setFixedSize(300, 300)
# 		#self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
#
#
#
#
# 		# # make expanded widget
# 		# self.expandedWidget = QtWidgets.QWidget(parent=self)
# 		# # bookend
# 		# self.firstBookendLabel = QtWidgets.QPushButton(
# 		# 	self.superItem.getBookendChars()[0],
# 		# 	parent=self.expandedWidget)
# 		# self.firstBookendLabel.clicked.connect(self.toggleExpanded)
# 		#
# 		# # initialise view
# 		# self.superViewType = superItem._getComponentTypeForObject(
# 		# 	self.superItem.wpPyObj, "view")
# 		# self.superView = self.superViewType(
# 		# 	self.superItem, parent=self.expandedWidget
# 		# )
# 		# #log("list widget init", self.superView, self.superItem.wpPyObj[0])
# 		# self.superView.buildIndexWidgets()
# 		# self.superView.resize(300, 300)
# 		# self.expandedWidget.resize(200, 200)
# 		#
# 		#
# 		# expandedLayout = QtWidgets.QHBoxLayout(self.expandedWidget)
# 		# expandedLayout.addWidget(self.firstBookendLabel)
# 		# expandedLayout.addWidget(self.superView)
# 		# expandedLayout.setAlignment(self.firstBookendLabel, QtCore.Qt.AlignTop)
# 		# self.expandedWidget.setLayout(expandedLayout)
# 		#
# 		# # make collapsed widget
# 		# # closed bookends (replace with typing summary)
# 		# self.stowedBookendLabel = QtWidgets.QPushButton(
# 		# 	self.superItem.getBookendChars()[0] + " ... " + self.superItem.getBookendChars()[1],
# 		# 	parent=self)
# 		# self.stowedBookendLabel.clicked.connect(self.toggleExpanded)
# 		#
# 		# # make stacked layout
# 		# # self.layout = QtWidgets.QStackedLayout(self)
# 		# # self.layout.setStackingMode(QtWidgets.QStackedLayout.StackAll)
# 		# # self.layout.addWidget(self.stowedBookendLabel)
# 		# # self.layout.addWidget(self.expandedWidget)
# 		# # self.setLayout(self.layout)
# 		# layout = QtWidgets.QVBoxLayout(self)
# 		# layout.addWidget(self.stowedBookendLabel)
# 		# layout.addWidget(self.expandedWidget)
# 		# self.setLayout(layout)
# 		#
# 		# layout.setAlignment(self.stowedBookendLabel, QtCore.Qt.AlignTop)
# 		# # layout.setStretch(0, 1)
# 		# # layout.setStretch(1, 1)
# 		#
# 		# # set model from item
# 		# self.superView.setModel(self.superItem.wpItemModel)
# 		#
# 		# # appearance
# 		# self.setAutoFillBackground(True)
#
#
#
# 	def isExpanded(self) -> bool:
# 		"""return True if expanded"""
# 		return self.expandedWidget.isVisible()
#
# 	def setExpanded(self, expanded:bool):
# 		"""set expanded state"""
# 		return
# 		self.expandedWidget.setVisible(expanded)
# 		self.stowedBookendLabel.setVisible(not expanded)
# 		#self.updateGeometry()
# 		#self.setGeometry(self.expandedWidget.geometry() if expanded else self.stowedBookendLabel.geometry())
# 		self.superView.scheduleDelayedItemsLayout()
# 		print("set expanded", expanded)
# 		self.superView.executeDelayedItemsLayout()
#
#
# 	def toggleExpanded(self, *args, **kwargs):
# 		"""toggle expanded state"""
# 		self.setExpanded(not self.isExpanded())


if __name__ == '__main__':

	from wpdex.ui.base import WpDexWindow
	data = [1, 2, 3,
	            [4, 5],
	        6,
	        [ 4, 5,
	          [ 6, 7, ],
	          8 ]
	        ]

	p = WpDexProxy(data)
	#ref = p.ref(3)
	ref = p.ref()
	log("ref", ref, "ref val", ref.rx.value)

	app = QtWidgets.QApplication()
	w = WpDexWindow(parent=None,
	                value=ref)
	w.show()
	app.exec_()


