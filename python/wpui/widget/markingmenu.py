
from __future__ import annotations

import random
import sys
from enum import IntEnum
from PySide2 import QtWidgets, QtCore, QtGui
import typing as T
from dataclasses import dataclass
from collections import defaultdict
from wplib.sequence import resolveSeqIndex, iterWindow, flatten

from wpui.widget.canvas.segment import RingSegment, SegmentData

"""honestly, past ed was cooking with this one

TODO: some way to scale the segments to get Mass Effect-style squished circles, 
always loved the look of those
"""

class MarkDirection(IntEnum):
	"""enum constants for what angle different options should appear
	TODO: replace with simple integers
	"""
	North = 0
	NorthEast = 1
	East = 2
	SouthEast = 3
	South = 4
	SouthWest = 5
	West = 6
	NorthWest = 7


directionAngleMap = {
	MarkDirection.North : 0,
	MarkDirection.NorthEast : 45,
	MarkDirection.East : 90,
	MarkDirection.SouthEast : 135,
	MarkDirection.South : 180,
	MarkDirection.SouthWest : 225,
	MarkDirection.West : 270,
	MarkDirection.NorthWest : 315
}

@dataclass
class MarkSegmentOption:
	text : str
	mouseOverText : str = ""
	direction : MarkDirection = None
	onMouseEnter : (T.Callable, MarkingMenu) = None
	onMouseRest : (T.Callable, MarkingMenu) = None
	onMouseRelease : (T.Callable, MarkingMenu) = None
	onMouseExit : (T.Callable, MarkingMenu) = None
	ringLayer : int = 0
	rgba : tuple = (0.5, 0.5, 0.5, 0.5)
	rgbaMouseOver : tuple = None
	pauseDelay : float = 0.3
	segmentData : SegmentData = None
	empty : bool = False


def startEndAnglesForItems(directions:list[MarkDirection],
                           ):
	"""return start and end angles of circle segment
	for each marking option
	each expands to polar midpoint between itself and neighbours

	for now we pass in margin angles here, which gives pizza-slice
	gaps instead of straight ones, but the alternative
	is complicated
	"""
	nSegments = len(directions)
	if nSegments == 1:
		return [(0, 360)]
	directions = sorted(directions)

	#print("sorted directions", directions)
	startEnds = []
	for i in range(nSegments):
		targetDir = directionAngleMap[directions[i]]
		nextDir = directionAngleMap[directions[resolveSeqIndex(i + 1, len(directions))]] or 360
		if nextDir < targetDir:
			nextDir = 360 + nextDir
		#print("targetdir", targetDir, "nextdir", nextDir)
		end = (targetDir + nextDir) / 2
		startEnds.append(end)
	#print("start ends", startEnds)
	return startEnds



class MarkingNode(QtWidgets.QGraphicsItem):
	def __init__(self, parent=None):
		super().__init__(parent)

class MarkingMenu:
	def __init__(self, items:list[MarkSegmentOption]):
		#self.items : list[MarkSegmentData] = []
		self.layerMap : dict[int, list[MarkSegmentOption]] = defaultdict(list)
		self.setItems(items)

	def setItems(self, items:list[MarkSegmentOption]):
		self.layerMap.clear()
		for i in items:
			self.layerMap[i.ringLayer].append(i)
		self.validateMenuItems(self.layerMap)
		self.buildSegmentDatas()

	def allItems(self)->list[MarkSegmentOption]:
		return flatten(self.layerMap.values())


	@staticmethod
	def validateMenuItems(layerMap):
		"""check that no items clash directions"""

		for k, v in layerMap.items():
			dirSet = set()
			for i in v:
				if i.direction in dirSet:
					raise RuntimeError("Clashing marking menu segment data : {}".format(v))

	def buildSegmentDatas(self, padding=3, innerRadius=40,
	                      layerWidth=40):
		nLayers = len(self.layerMap.keys())
		layerStep = 0.5 / nLayers
		layerStep = layerWidth
		totalRad = innerRadius + layerWidth * nLayers
		for layer, v in self.layerMap.items():
			#print("items", v)
			angles = startEndAnglesForItems([i.direction for i in v])
			#print("angles", angles)
			innerRadFract = (innerRadius + layerWidth * (layer) + padding) / totalRad
			outerRadFract = (innerRadius + layerWidth * (layer + 1) ) / totalRad

			for i, (prev, current, next) in enumerate(iterWindow(angles)):
				if next < current:
					current = current - 360
				segData = SegmentData(current, next,
				                      innerRadiusFract= innerRadFract,
				                      outerRadiusFract= outerRadFract,
				                      maxRadius=totalRad,
				                      marginAngle=padding)
				v[i].segmentData = segData




"""qt display for marking menu is made as a separate 
transparent graphics that gets shown as an overlay
on top of what ever base widget owns the menu"""

class MarkSegment(RingSegment):
	mouseEntered = QtCore.Signal()
	mouseRested = QtCore.Signal()
	mouseReleased = QtCore.Signal()
	mouseLeft = QtCore.Signal()
	
	def __init__(self,
	             markData:MarkSegmentOption,
	             parent=None):
		segData = markData.segmentData
		super(MarkSegment, self).__init__(segData,
	             parent=parent
		)
		self.markData = markData
		self.textItem = QtWidgets.QGraphicsTextItem(self.markData.text, parent=self)
		#print("segData", segData)
		self.textItem.setDefaultTextColor(QtGui.Qt.white)


		self.path = self.outlinePath()
		self.setBoundingRegionGranularity(0.25)
		self._prevUnderMouse = False # have to track this manually to check for entry and exit
		self.setFlag(self.ItemIsSelectable, True)
		self.setFlag(self.ItemClipsToShape, True)

	def __eq__(self, other):

		if other is None:
			return False
		assert isinstance(other, MarkSegment)
		return self.markData == other.markData

	def boundingRect(self) -> QtCore.QRectF:
		return self.path.boundingRect()

	def boundingRegion(self, itemToDeviceTransform:QtGui.QTransform) -> QtGui.QRegion:
		return QtGui.QRegion(self.path.toFillPolygon())

	def updatePath(self):
		self.mainPoints()
		self.outlinePath()

		# place text
		self.textItem.setPos(self.centrePath().pointAtPercent(0.5) -
		                     self.textItem.boundingRect().bottomRight()/2)

	def shape(self) -> PySide2.QtGui.QPainterPath:
		return self.path

	def paint(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionGraphicsItem, widget:T.Optional[QtWidgets.QWidget]=...) -> None:
		"""colour in segment,
		change colour if mouse is over it"""
		#print("paint")
		random.seed()
		fillCol = QtGui.QColor.fromRgbF(*self.markData.rgba)
		mouseOverFillCol = QtGui.QColor.fromRgbF(*(self.markData.rgbaMouseOver
			if self.markData.rgbaMouseOver else fillCol.lighter(40)))

		#print("under mouse", self.isUnderMouse())
		if self.isUnderMouse(): # emit mouse enter / exit events
			if not self._prevUnderMouse:
				#self.mouseEntered.emit()
				pass

			brush = QtGui.QBrush(mouseOverFillCol)
		else:
			if self._prevUnderMouse:
				#self.mouseLeft.emit()
				pass
			brush = QtGui.QBrush(fillCol)
		self._prevUnderMouse = self.isUnderMouse()
		path = (self.path)
		painter.fillPath(path, brush)

		#print("path points", self._pathPoints)
		painter.setBrush(QtCore.Qt.green)
		painter.drawPolygon(QtGui.QPolygon(self._pathPoints))

		# draw text
		# drawTextAlongPath(self.markData.text,
		#                   self.centrePath(),
		#                   painter)



def _onOuterAMouseOver():
	print("outerA mouse over")

def _mouseRelease():
	print("option mouse release")

def buildTestMenu()->MarkingMenu:


	# outer level - shamrock
	leafA = MarkSegmentOption("leafA", direction=MarkDirection.NorthEast)
	leafB = MarkSegmentOption("", direction=MarkDirection.NorthEast, empty=True)
	leafC = MarkSegmentOption("leafC", direction=MarkDirection.East)
	leafD = MarkSegmentOption("", direction=MarkDirection.SouthEast, empty=True)
	leafE = MarkSegmentOption("leafE", direction=MarkDirection.South)
	leafF = MarkSegmentOption("", direction=MarkDirection.SouthWest, empty=True)
	leafG = MarkSegmentOption("leafG", direction=MarkDirection.West)
	leafH = MarkSegmentOption("", direction=MarkDirection.NorthWest, empty=True)

	leafMenu = MarkingMenu([leafA, leafB, leafC, leafD, leafE, leafF, leafG, leafH])


	# mid level
	outerSubOptionA = MarkSegmentOption("outerOptionA",
	                                    direction=MarkDirection.North,
	                                    )
	outerSubOptionB = MarkSegmentOption("outerOptionB",
	                                    direction=MarkDirection.South,
	                                    onMouseEnter=leafMenu
	                                    )
	outerMenu = MarkingMenu([outerSubOptionA, outerSubOptionB])
	#
	outerA = MarkSegmentOption("outerOptionA",
	                           direction=MarkDirection.West,
	                           onMouseEnter=outerMenu,
	                           ringLayer=1
	                           )
	outerB = MarkSegmentOption("outerOptionB",
	                           direction=MarkDirection.East,
	                           onMouseEnter=outerMenu,
	                           ringLayer=1
	                           )
	innerA = MarkSegmentOption("innerA",
	                           direction=MarkDirection.NorthEast,
	                           onMouseRelease=_mouseRelease
	                           )
	innerB = MarkSegmentOption("innerB",
	                           direction=MarkDirection.East,
	                           onMouseRelease=_mouseRelease
	                           )
	innerC = MarkSegmentOption("innerC",
	                           direction=MarkDirection.SouthEast,
	                           onMouseRelease=_mouseRelease
	                           )
	innerD = MarkSegmentOption("innerD",
	                           direction=MarkDirection.West,
	                           onMouseRelease=_mouseRelease,

	                           )

	mainMenu = MarkingMenu([innerA, innerB, innerC, innerD, outerA, outerB])
	#mainMenu = MarkingMenu([innerC, innerD])
	return mainMenu



class MarkingMenuScene(QtWidgets.QGraphicsScene):
	pass
	# def backgroundBrush(self) -> PySide2.QtGui.QBrush:
	# 	backBrush = QtGui.QBrush(QtGui.QColor(200, 200, 0, 100))
	# 	return backBrush
	#
	# def drawBackground(self, painter:PySide2.QtGui.QPainter, rect:PySide2.QtCore.QRectF) -> None:
	# 	return
	# 	# painter.setBrush(self.backgroundBrush())
	# 	# painter.drawRect(rect)
	# 	pass


class MarkingMenuView(QtWidgets.QGraphicsView):
	"""main class for displaying marking menu, scene is
	directly contained here"""

	def __init__(self, parent=None):
		super(MarkingMenuView, self).__init__(parent)
		self._markScene = MarkingMenuScene(self.parent())
		self.menu : MarkingMenu = None
		self.setScene(self._markScene)
		self.marginAngle = 5
		self.setRenderHint(QtGui.QPainter.Antialiasing, True)
		self.installEventFilter(self)
		self.mousePos = QtCore.QPoint()

		self.prevPointRad = 5

		self.positionStack = [] # path of positions taken by cursor
		self.menuStack : list[MarkingMenu] = [] # list of menus currently DISPLAYED
		self.mouseLineItem : QtWidgets.QGraphicsLineItem = None # final segment from point to cursor
		#self.setMouseTracking(True)
		self.setAlignment(QtCore.Qt.AlignTop)
		self.scene().setSceneRect(self.rect())

		self._prevSegUnderMouse : MarkSegment = None # detect enter / exit mouse events for shapes

		# self.setWindowFlags(QtCore.Qt.Widget | QtCore.Qt.FramelessWindowHint)
		# self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
		# self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
		#self.setAttribute(QtCore.Qt.WA_PaintOnScreen, True)
		#
		#
		#self.setWindowOpacity(0)
		#self.setAutoFillBackground(True)
		#self.setAutoFillBackground(False)


		#self.setBackgroundRole(QtGui.QPalette.Window)
		#self.setBackgroundRole(QtGui.QPalette.NoRole)
		self.setStyleSheet("background:transparent")
		# this fucking line is the only thing that actually makes a widget transparent

	# def drawBackground(self, painter:PySide2.QtGui.QPainter, rect:PySide2.QtCore.QRectF) -> None:
	# 	print("view draw background")
	# 	painter.setBrush(self.backgroundBrush())
	# 	painter.drawRect(rect)
	# 	return
	# # 	pass
	#
	def backgroundBrush(self) -> PySide2.QtGui.QBrush:
		return QtGui.QBrush(QtGui.QColor(0, 0, 200, 100))


	def mouseLine(self)->QtCore.QLineF:
		return QtCore.QLineF(self.positionStack[-1], QtCore.QPointF(self.mousePos))

	def resizeEvent(self, event:PySide2.QtGui.QResizeEvent) -> None:
		super(MarkingMenuView, self).resizeEvent(event)
		self.scene().setSceneRect(self.rect())


	def eventFilter(self, arg__1:QtCore.QObject, arg__2:QtCore.QEvent) -> bool:
		if arg__2.type() == QtCore.QEvent.MouseMove:
			self.mouseMoveEvent(arg__2)
			self.scene().setSceneRect(self.rect())
			return True

		#print("view event filter", arg__1, arg__2)
		return False
	
	def mouseReleaseEvent(self, event:PySide2.QtGui.QMouseEvent) -> None:
		super(MarkingMenuView, self).mouseReleaseEvent(event)
		print("mark mouseReleaseEvent", event)
		if self._prevSegUnderMouse:
			self.onSegmentMouseReleased(self._prevSegUnderMouse)
		self.clearMenus()

	def mouseMoveEvent(self, event:QtGui.QMouseEvent) -> None:
		self.scene().update()
		super(MarkingMenuView, self).mouseMoveEvent(event)
		self.mousePos = event.pos()
		if self.mouseLineItem:
			self.mouseLineItem.setLine(
				self.mouseLine())


		# # text
		# segs = [i for i in self.scene().items() if isinstance( i, MarkSegment)]
		# drawTextAlongPath(segs[0].markData.text,
		#                   segs[0].centrePath(),
		#                   QtGui.QPainter(self))

		# check for enter / exit events on shape
		segsUnderMouse = [i for i in self.scene().items(event.pos()) if isinstance(i, MarkSegment)]
		#print("under mouse", segsUnderMouse)
		if not segsUnderMouse:
			if self._prevSegUnderMouse:
				self.onSegmentMouseExited(self._prevSegUnderMouse)
			self._prevSegUnderMouse = None
			return
		if segsUnderMouse[0] != self._prevSegUnderMouse:
			if self._prevSegUnderMouse:
				self.onSegmentMouseExited(self._prevSegUnderMouse)
			self._prevSegUnderMouse = segsUnderMouse[0]
			self.onSegmentMouseEntered(self._prevSegUnderMouse)

	def evalMenuEffects(self, markDataVal:(callable, tuple[callable]), allowSubMenu=True):
		if isinstance(markDataVal, (tuple, list, set)):
			for i in markDataVal:
				self.evalMenuEffects(i, allowSubMenu)
			return
		if callable(markDataVal):
			markDataVal()
		if isinstance(markDataVal, MarkingMenu) and allowSubMenu:
			self.pushMenu(markDataVal, self.mousePos)


	def onSegmentMouseReleased(self, segment:MarkSegment):
		self.evalMenuEffects(segment.markData.onMouseRelease, allowSubMenu=False)

	def onSegmentMouseRested(self, segment:MarkSegment):
		"""check if a new set of menu options has to pop up"""
		self.evalMenuEffects(segment.markData.onMouseRest, allowSubMenu=True)

	def onSegmentMouseEntered(self, segment:MarkSegment):
		#print("segment entered", segment)
		self.evalMenuEffects(segment.markData.onMouseEnter, allowSubMenu=True)

	def onSegmentMouseExited(self, segment:MarkSegment):
		self.evalMenuEffects(segment.markData.onMouseExit, allowSubMenu=False)
		pass

	def graphicsItemsForMarkingMenu(self, menu:MarkingMenu, position:QtCore.QPoint)->list[QtWidgets.QGraphicsItem]:
		items = []
		null = QtWidgets.QGraphicsItemGroup()
		for layer, segments in menu.layerMap.items():
			for seg in segments:
				if seg.empty: continue
				markSeg = MarkSegment(seg, parent=null)
		items = [null]
		null.setPos(position)
		for i in null.childItems():
			if isinstance(i, MarkSegment):
				i.updatePath()
		return items


	def refreshGraphics(self):
		self.scene().clear()
		prevPointColour = QtCore.Qt.gray

		if self.menuStack:
			items = self.graphicsItemsForMarkingMenu(self.menuStack[-1], self.positionStack[-1])
			for i in items:
				self.scene().addItem(i)

		for position, menu in zip(self.positionStack, self.menuStack):

			position = QtCore.QPointF(position) - QtCore.QPointF(self.prevPointRad, self.prevPointRad) / 2

			posEllipse = QtWidgets.QGraphicsEllipseItem(
				position.x(), position.y(),
				self.prevPointRad, self.prevPointRad
			)
			posEllipse.setBrush(prevPointColour)
			posEllipse.setPen(QtGui.QPen(QtCore.Qt.lightGray, self.prevPointRad / 4))
			self.scene().addItem(posEllipse)

			cursorPath = QtGui.QPainterPath()
			cursorPath.moveTo(self.positionStack[0])
			#print("position stack", self.positionStack)

			prevPen = QtGui.QPen(prevPointColour, 3)
			for i in range(len(self.positionStack[:-1])):
				prevLineItem = QtWidgets.QGraphicsLineItem(
					QtCore.QLineF(self.positionStack[i], self.positionStack[i + 1]))
				prevLineItem.setPen(prevPen)
				self.scene().addItem(prevLineItem)

			self.mouseLineItem = QtWidgets.QGraphicsLineItem(self.mouseLine())
			self.mouseLineItem.setPen(QtGui.QPen(prevPointColour, 3))
			self.scene().addItem(self.mouseLineItem)


		self.scene().update()

	def pushMenu(self, menu:MarkingMenu, pos:QtCore.QPoint):
		# self.scene().clear()
		#print("push menu")
		# build menu items
		self.mousePos = pos
		self.menuStack.append(menu)
		self.positionStack.append(self.mapToScene(pos))
		self.refreshGraphics()

	def popMenu(self):
		"""removes the most recent displayed menu"""
		self.positionStack.pop(-1)
		self.menuStack.pop(-1)
		self.refreshGraphics()

	def clearMenus(self):
		while self.positionStack:
			self.popMenu()
		self.scene().clear()
		#self.mousePos = None
		pass


class MarkMenuWidgetMixin:
	"""widget mixin to create a view overlay for marking menu,
	and function to show menu at a given position"""
	def __init__(self, showButtonAndModifier=(QtCore.Qt.RightButton, None)):
		self.markView = MarkingMenuView(parent=self)
		self.markView.hide()
		self.showButton = showButtonAndModifier[0]
		self.showModifier = showButtonAndModifier[1]

	def showMarkView(self, state=True):
		"""handles showing graphicsView overlay widget for marking menu"""
		self.markView.setGeometry( self.geometry())
		self.markView.scene().setSceneRect(self.rect())
		if state:
			self.markView.show()
			self.markView.raise_()
			self.markView.setMouseTracking(True)
			self.markView.grabMouse()
			self.markView.setFocus()
		else:
			self.markView.hide()
			self.markView.clearMenus()
			self.markView.setMouseTracking(False)
			self.markView.releaseMouse()

	def getMarkingMenu(self)->MarkingMenu:
		"""reimplement to actually build the marking menu for this widget class"""
		raise NotImplementedError

	def mousePressEvent(self, event:QtGui.QMouseEvent) -> None:
		"""pass this the event gathered from the normal widget event
		processing"""
		# if self.showModifier:
		# 	if self.showModifier != event.modifiers()

		# ignore the event unless it matches the desired button
		if event.button() != self.showButton:
			return

		menu = self.getMarkingMenu()
		self.markView.pushMenu(menu, event.pos())
		self.showMarkView(True)

	def mouseReleaseEvent(self, event:QtGui.QMouseEvent) -> None:
		# directly pass event to view, not sure why this one doesn't go otherwise
		self.markView.mouseReleaseEvent(event)
		self.showMarkView(False)


class TestWindow(QtWidgets.QWidget, MarkMenuWidgetMixin):

	def __init__(self, parent=None):
		QtWidgets.QWidget.__init__(self, parent)
		MarkMenuWidgetMixin.__init__(self)

		self.markWindowWidth = 200
		pass

	def buildMarkingMenu(self):
		return buildTestMenu()

	def mousePressEvent(self, event:QtGui.QMouseEvent) -> None:
		MarkMenuWidgetMixin.mousePressEvent(self, event)

	def mouseReleaseEvent(self, event:QtGui.QMouseEvent) -> None:
		MarkMenuWidgetMixin.mouseReleaseEvent(self, event)





if __name__ == '__main__':

	app = QtWidgets.QApplication(sys.argv)
	win = QtWidgets.QMainWindow()
	widg = TestWindow(win)
	win.setCentralWidget(widg)
	win.show()
	sys.exit(app.exec_())