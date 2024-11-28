
from __future__ import annotations
import types
from wplib import log


from PySide2 import QtCore, QtWidgets, QtGui
from wptree import Tree

from wplib import inheritance
from wpui.canvas import *
from wpdex.ui.atomic import ExpWidget
from wpdex import WpDexProxy, WX, react
from wpdex.ui import AtomicUiInterface
from wpui.widget.collapsible import ShrinkWrapWidget


if T.TYPE_CHECKING:
	from .node import NodeDelegate



PLUG_TREE_INDENT = 20

def paintDropdownSquare(rect:QtCore.QRect, painter:QtGui.QPainter,
          canExpand=True, expanded=False):
	painter.drawRect(rect)
	innerRect = QtCore.QRect(rect)
	innerRect.setSize(rect.size() / 3.0 )
	innerRect.moveCenter(rect.center())
	if expanded:
		rect.moveCenter(QtCore.QPoint(rect.center().x(), rect.bottom()))
	if canExpand:
		painter.drawRect(innerRect)


class OpenPlugRowConnectorPanel(QtWidgets.QGraphicsLineItem):
	"""single line that goes on the end of an open branch, showing actual connections
	from all node attributes

	should try and relax multiple connections to individual incoming streams -
	LATER
	"""

	def __init__(self, parent:OpenPlugRow=None):
		QtWidgets.QGraphicsLineItem.__init__(self, parent=parent)


class OpenPlugRow(QtWidgets.QGraphicsItem):
	""" a single triple of
	( attrRef (s), attribute, path )
	including connector panel on end
	TODO: in future we might just make this a single expression
	"""
	if T.TYPE_CHECKING:
		def parentItem(self)->PlugBranchItem: ...
	def __init__(self,
	             valueList:list[str, str, list],
	             parent:PlugBranchItem=None):
		QtWidgets.QGraphicsItem.__init__(self, parent)
		self.proxyW = QtWidgets.QGraphicsProxyWidget(parent=self
		                                             )
		self.w = ShrinkWrapWidget(parent=None)
		self.w.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
		self.w.setAutoFillBackground(False)
		self.w.setAttribute(QtCore.Qt.WA_TranslucentBackground)
		self.setWidgetResult = self.proxyW.setWidget(self.w)

		self.nodeLine = ExpWidget(value=valueList[0],
		                          parent=self.w)
		self.attrLine = ExpWidget(value=valueList[1],
		                          parent=self.w)
		self.pathLine = ExpWidget(value=valueList[2],
		                          parent=self.w)
		self.w.setLayout(QtWidgets.QHBoxLayout(self.w))
		for i in (self.nodeLine, self.attrLine, self.pathLine):
			self.w.layout().addWidget(i)

	def isInput(self)->bool:
		return self.parentItem().isInput

	def paint(self, painter, option, widget=...):
		pass

	def boundingRect(self):
		return self.childrenBoundingRect()

# base = (WpCanvasElement,
#         ConnectionPoint,
#         QtWidgets.QGraphicsItem)
# if T.TYPE_CHECKING:
# 	base = (QtWidgets.QGraphicsItem,
# 	        ConnectionPoint,
# 	        WpCanvasElement)
class ChimaeraConnector(
	ConnectionPoint,
	QtWidgets.QGraphicsItem,
):
	"""
	Connection point for data streams - when open, show a ]
	shape to show available; when attached, show a [ around wire

	TODO: add different colours for data types -
		typed branches in template tree output
		for now just light grey
	"""

	def __init__(self, parent=None,
	             size=20,
	             isInput=True):
		QtWidgets.QGraphicsItem.__init__(self, parent=parent)
		WpCanvasElement.__init__(self, obj=None)
		ConnectionPoint.__init__(self,
		                         isInput=isInput,
		                         )
		self.size = size


	def boundingRect(self):
		pad = 50
		return QtCore.QRectF(self.size, self.size,
		                     self.size, self.size).marginsAdded(
			QtCore.QMarginsF(pad, pad, pad, pad)
		)

	def connectionPoint(self,
	                    forDelegate:ConnectionGroupDelegate=None
	                    ) ->tuple[tuple[float, float], (tuple[float, float], None)]:
		pos = (self.size / 2.0, self.size / 2.0)
		if self.isInput: # incoming
			vec = (-1, 0)
		else:
			vec = (1, 0) # outgoing
		return pos, vec

	def paint(self, painter:QtGui.QPainter, option, widget=...):
		"""if plug is totally open, draw open bracket
		else, draw bracket arms closed around line endpoint

		draw lines on a 5x5 grid, then scale up
		"""
		painter.save()
		pen = QtGui.QPen(QtCore.Qt.lightGray)
		pen.setWidthF(0.5)
		painter.setPen(pen)
		scaleF = self.size / 5.0
		painter.scale(scaleF, scaleF)

		painter.setBrush(QtCore.Qt.NoBrush)

		if self.isUnderMouse():
			painter.scale(1.2, 1.2)
		if self.isInput:
			lines = self.connectionLines()
			if lines:  # has connections
				# points sorted properly, [ (x, y) ]
				upLinePoints = [(1, 0), (0, 0), (0, 1)]
				downLinePoints = [(1, 4), (0, 4), (0, 3)]
			else: # empty
				upLinePoints = [(4, 1), (4, 0), (3, 0)]
				downLinePoints = [(4, 3), (4, 4), (3, 4)]
			upLineQPoints = [QtCore.QPointF(*i) for i in upLinePoints]
			downLineQPoints = [QtCore.QPointF(*i) for i in downLinePoints]

			painter.drawLine(upLineQPoints[0], upLineQPoints[1])
			painter.drawLine(upLineQPoints[1], upLineQPoints[2])
			painter.drawLine(downLineQPoints[0], downLineQPoints[1])
			painter.drawLine(downLineQPoints[1], downLineQPoints[2])
		else: # for outputs just draw a square
			#painter.drawRect(QtCore.QRect(-2, -2, 2, 2))
			painter.drawRect(QtCore.QRect(0, 0, 3, 3))

		painter.restore()




class PlugBranchItem(QtWidgets.QGraphicsItem,
                     AtomicUiInterface,
                     metaclass=inheritance.resolveInheritedMetaClass(
	                     QtWidgets.QGraphicsItem, AtomicUiInterface
                     )):
	"""single branch of tree for connections, showing branch name
	either collapsed or expanded view of individual link items

	testing using the same reactive interface in this too, gaining
	more faith in it as a general solution

	for now don't allow direct editing of main tree structure

	on dragging wire to plug, by default replace entry
	if hold shift, add a new tie for this connection

	NO EXPANDING FOR NOW

	"""

	if T.TYPE_CHECKING:
		def parentItem(self)->(PlugBranchItem, NodeDelegate):...
		def value(self) ->Tree:...

	def __init__(self,
	             value: Tree|WpDexProxy|WX,
	             parent=None,
	             isInput=True,

	             ):
		AtomicUiInterface.__init__(self,
		                           value=value)
		QtWidgets.QGraphicsItem.__init__(self, parent=parent)

		#self.branch = branch
		self.isInput = isInput
		self.branchItems : dict[Tree, PlugBranchItem] = {}
		self.rowItems : list[OpenPlugRow] = []
		self.lines = []
		self.connector = ChimaeraConnector(parent=self,
		                                   isInput=self.isInput
		                                   )

		self.text = QtWidgets.QGraphicsTextItem(self.value().name,
		                                        parent=self)
		self.text.setDefaultTextColor(QtCore.Qt.lightGray)
		self.text.setPos(0, -10)
		if self.isInput:
			self.connector.setPos(self.boundingRect().left() - 20, 0)
		else:
			self.connector.setPos(self.text.pos().x() + self.text.boundingRect().width() + 20, 0)
	def boundingRect(self):
		return self.text.boundingRect().marginsAdded(
			QtCore.QMarginsF(5, 5, 5, 5)
		)
		#return QtCore.QRectF(0, 0, 100, 100)
		# return self.childrenBoundingRect().marginsAdded(
		# 	QtCore.QMarginsF(5, 5, 5, 5)
		# )

	def paint(self, painter:QtGui.QPainter, option, widget=...):
		painter.save()
		painter.setPen(QtCore.Qt.darkGray)
		painter.drawLine(self.boundingRect().left(), 10,
		                 self.boundingRect().right(), 10)

		painter.restore()

	def isOpen(self):
		"""return display state of this branch, if individual
		ties are visible for each connection made"""

	def nodeDelegate(self)->NodeDelegate:
		p = self.parentItem()
		while not isinstance(p, NodeDelegate):
			p = p.parentItem()
		return p

	def syncBranchItems(self):
		"""build out child items - leave layout for separate method
		"""
		for i in self.branchItems.values():
			self.scene().removeItem(i)
		self.branchItems.clear()

		for k, b in self.value().branchMap():
			self.branchItems[b] = PlugBranchItem(self.valueProxy().ref(k),
			                                     parent=self,
			                                     isInput=self.isInput)

	def syncRowItems(self):
		"""look at the current value of the tree, create an entry for each existing list
		item,
		plus a trailing empty one"""
		for i in self.rowItems:
			self.scene().removeItem(i)
		self.rowItems.clear()

		branch = self.value()
		assert isinstance(branch, Tree)
		if not isinstance(branch.value, list):
			assert not branch.value, f"Linking tree {branch} has non-list, non-None value"
			# set value from within reactive slot - SHOULD be ok as long as it doesn't loop
			branch.value = [("", "@F", "")]
			# this should trigger a redraw anyway
			return
			# unsure how to make this reactive - hook into empty and default values on dex
		# create a tie for each one
		for i, tie in enumerate(branch.value):
			self.rowItems.append(
				OpenPlugRow(tie, parent=self)
			)

	# for reactive widgets, allow an empty entry?
	# allow buttons to add and remove empty entries?
	# set this on WPDEX? default trailing value, default empty value?

	def syncLayout(self):
		"""only single layer
		TODO: extend to open plug entry items
		"""
		if self.isInput:
			self.connector.setPos(-20, 0)

		else:
			self.connector.setPos(self.text.boundingRect().width() + self.text.pos().x() + 20, 0)

		totalRect = self.boundingRect().united(self.childrenBoundingRect())
		y = totalRect.height()
		for i in self.childItems():
			if not isinstance(i, PlugBranchItem):
				continue

			i.setPos(PLUG_TREE_INDENT, y)
			iRect = i.boundingRect().united(i.childrenBoundingRect())
			y += iRect.height()

	def syncLines(self):
		pass

	def _syncUiFromValue(self, *args, **kwargs):
		"""
		main reactive UI update slot
		rebuild items and sync layout"""
		self.syncBranchItems()
		#self.syncRowItems() #TODO
		self.syncLayout()
		self.syncLines()




class TreePlugSpine(QtWidgets.QGraphicsItem):
	"""draw an expanding tree structure -
	this is used to display connections coming in to the
	"linking" tree of each attribute
	"""

	def __init__(self, parent=None, size=10.0):
		super().__init__(parent)
		self._expanded = False
		self.size = 10.0

	def childTreeSpines(self)->list[TreePlugSpine]:
		return [i for i in self.childItems() if isinstance(i, TreePlugSpine)]

	def expanded(self):
		return self._expanded
	def setExpanded(self, state):
		self._expanded = state
		self.update()

	def boundingRect(self):
		return QtCore.QRectF(0, 0, self.size, self.size)
	def paint(self, painter:QtGui.QPainter, option, widget=...):
		"""draw a line backwards,
		a cross if not expanded,
		and the vertical bar down if expanded
		"""
		mid = self.size / 2.0
		painter.drawLine(-mid, mid, mid, mid)
		childSpines = self.childTreeSpines()
		if not self.childTreeSpines(): return
		paintDropdownSquare(rect=self.boundingRect(), painter=painter,
		                    canExpand=True, expanded=self.expanded())


if __name__ == '__main__':
	app = QtWidgets.QApplication()
	obj = QtCore.QObject()
	d = {"a" : "v"}
	d[obj] = "test"
	print(d[obj])
	print(d)
	print(d.keys())
	#print(d[hash(obj)])
	item = QtWidgets.QGraphicsRectItem()
	d[item] = "itemVal"
	print(d)
	print(d[item])
	print(d[QtCore.QObject()])
	app.exec_()
