
from __future__ import annotations
import typing as T

import numpy as np
import pathlib

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.sequence import toSeq
from wplib.object import TypeNamespace, Signal

import subprocess

expandingPolicy = QtWidgets.QSizePolicy(
	QtWidgets.QSizePolicy.Expanding,
	QtWidgets.QSizePolicy.Expanding,
)

fixedPolicy = QtWidgets.QSizePolicy(
	QtWidgets.QSizePolicy.Fixed,
	QtWidgets.QSizePolicy.Fixed,
)
shrinkingPolicy = QtWidgets.QSizePolicy(
	QtWidgets.QSizePolicy.Maximum,
	QtWidgets.QSizePolicy.Fixed,
)

def openExplorerOnPath(path:(pathlib.Path, str),
                       lowestExisting=True):
	"""opens a new process of windows explorer,
	focused on given path
	"""
	p = pathlib.Path(path)
	if not p.exists():
		if lowestExisting:
			while not p.exists():
				try: p = p.parents[0]
				except IndexError:
					raise FileNotFoundError(f"No existing path to open explorer for any part of path {path}, {type(path)} ")
		else:
			raise FileNotFoundError(f"No existing path to open explorer for path {path}, {type(path)} ")
	if p.is_file():
		return subprocess.Popen(f'explorer /select,"{p}"')
	elif p.is_dir():
		return subprocess.Popen(f'explorer "{p}"')

def widgetParents(w:QtWidgets.QWidget):
	result = []
	while w.parentWidget():
		result.append(w.parentWidget())
		w = w.parentWidget()
	return result

def rootWidget(w:QtWidgets.QWidget):
	while w.parentWidget():
		w = w.parentWidget()
	return w

def widgetChildMap(w:QtWidgets.QWidget, includeObjects=True, onlyNamed=True)->dict[str, (QtWidgets.QWidget, QtCore.QObject)]:
	result = {}
	for i in w.children():
		if not isinstance(i, QtWidgets.QWidget):
			if not includeObjects:
				continue
		if not str(i.objectName()):
			if not onlyNamed:
				continue
		result[i.objectName()] = i
	return result

class muteQtSignals:
	"""small context for muting qt signals around a block"""
	def __init__(self, obj:QtCore.QObject):
		self.objs = toSeq(obj)
	def __enter__(self):
		for i in self.objs:
			i.blockSignals(True)
	def __exit__(self, exc_type, exc_val, exc_tb):
		for i in self.objs:
			i.blockSignals(False)

def arrToQMatrix(arr:np.ndarray)->QtGui.QMatrix:
	"""convert numpy array to QMatrix"""
	return QtGui.QMatrix(arr[0, 0], arr[0, 1], arr[1, 0], arr[1, 1], arr[0, 2], arr[1, 2])

def qmatrixToArr(mat:QtGui.QMatrix)->np.ndarray:
	"""convert QMatrix to numpy array"""
	return np.array([[mat.m11(), mat.m12(), mat.dx()],
	                 [mat.m21(), mat.m22(), mat.dy()],
	                 [0, 0, 1]])

def qTransformToArr(mat:QtGui.QTransform)->np.ndarray:
	"""convert QMatrix to numpy array"""
	return np.array([[mat.m11(), mat.m12(), mat.dx()],
	                 [mat.m21(), mat.m22(), mat.dy()],
	                 [0, 0, 1]])

def qRectToArr(rect:(QtCore.QRect, QtCore.QRectF),
               originSize=True)->np.ndarray:
	dtype = float
	if isinstance(rect, QtCore.QRect):
		dtype = int
	if originSize:
		arr = np.array([rect.topLeft().toTuple(), rect.size().toTuple()], dtype=dtype)
	else:
		arr = np.array([rect.topLeft().toTuple(), rect.bottomRight().toTuple()], dtype=dtype)
	return arr

class Direction(TypeNamespace):
	"""use for arrow directions,
	widget placement, etc

	TODO: move this higher
	"""

	class _Base (TypeNamespace.base()):
		direction = (0, 0)
		angle = 0
		pass

	class Left(_Base):
		direction = (-1, 0)
		angle = 0
	class Up(_Base):
		direction = (0, 1)
		angle = -90
	class Right(_Base):
		direction = (1, 0)
		angle = -180
	class Down(_Base):
		direction = (0, -1)
		angle = -270

def tripoints():
	return [QtCore.QPointF(0, 0),
	        QtCore.QPointF(1, 0.5),
	        QtCore.QPointF(0, 1),
	        ]


class UiIcon(TypeNamespace):
	"""holder for common symbols to use throughout ui,
	providing ways to draw in widgets, paint in graphics items etc
	"""

	#class _Base(QtWidgets.QGraphicsItem, TypeNamespace.base()):
	class _Base(TypeNamespace.base()):
		"""smallest possible base class for   common shapes
		within uniform 0-1 square area"""

		def __init__(self, **kwargs ):
			#QtWidgets.QGraphicsItem.__init__(self, parent=parent)
			#GraphicsMouseSignalMixin.__init__(self)
			# self._pen = self.defaultPen()
			# self._brush = self.defaultBrush()
			self._initKwargs = kwargs

		def defaultPen(self)->QtGui.QPen:
			pen = QtGui.QPen(QtGui.QColor(200, 200, 200))
			pen.setWidthF(0.1)
			return pen

		def defaultBrush(self)->QtGui.QBrush:
			return QtGui.QBrush(QtGui.QColor(128, 128, 128))

		# def setPen(self, pen :QtGui.QPen):
		# 	self._pen = pen
		# 	self.update()

		# def setBrush(self, brush :QtGui.QBrush):
		# 	self._brush = brush
		# 	self.update()

		def boundingRect(self)->QtCore.QRectF:
			return QtCore.QRectF(0, 0, 1, 1)

		def drawIcon(self, painter :QtGui.QPainter, rect :QtCore.QRectF,
		             initKwargs:T.Optional[T.Dict[str, T.Any]]=None) -> None:
			"""draw icon within rect"""
			raise NotImplementedError

		def paint(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionGraphicsItem, widget:T.Optional[QtWidgets.QWidget]=...) -> None:
			"""draw the icon"""
			painter.save()
			painter.setPen(self._pen)
			painter.setBrush(self._brush)
			self.drawIcon(painter, QtCore.QRectF(
				0, 0, 1, 1
			), self._initKwargs)
			painter.restore()

		def mouseMoveEvent(self, event:QtWidgets.QGraphicsSceneMouseEvent) -> None:
			#print('icon move')
			#QtWidgets.QGraphicsItem.mouseMoveEvent(self, event)
			#event.accept()
			pass


	class Arrow(_Base):
		"""draws arrow icon"""

		def __init__(self, direction=Direction.Left, parent=None, **kwargs):
			UiIcon._Base.__init__(self, parent=parent, direction=direction, **kwargs)

		def drawIcon(self, painter :QtGui.QPainter, rect :QtCore.QRectF,
		             initKwargs:T.Optional[T.Dict[str, T.Any]]=None) -> None:
			"""draw icon within rect"""
			direction : Direction.T() = self._initKwargs.get('direction', Direction.Left)
			painter.translate(QtCore.QPointF(0.5, 0.5))
			painter.rotate(direction.angle)
			painter.translate(QtCore.QPointF(-0.5, -0.5))
			polygon = QtGui.QPolygonF(tripoints())

			painter.drawPolygon(polygon)

	class Circle(_Base):
		"""draws circle icon"""

		def drawIcon(self, painter :QtGui.QPainter, rect :QtCore.QRectF,
		             initKwargs:T.Optional[T.Dict[str, T.Any]]=None) -> None:
			"""draw icon within rect"""
			painter.drawEllipse(rect)

	class Square(_Base):
		"""draws square icon"""

		def drawIcon(self, painter :QtGui.QPainter, rect :QtCore.QRectF,
		             initKwargs:T.Optional[T.Dict[str, T.Any]]=None) -> None:
			"""draw icon within rect"""
			painter.drawRect(rect)

	class X(_Base):
		"""draws X icon"""

		def drawIcon(self, painter :QtGui.QPainter, rect :QtCore.QRectF,
		             initKwargs:T.Optional[T.Dict[str, T.Any]]=None) -> None:
			"""draw icon within rect"""
			painter.drawLine(rect.topLeft(), rect.bottomRight())
			painter.drawLine(rect.topRight(), rect.bottomLeft())

	class Plus(_Base):
		"""draws + icon"""

		def drawIcon(self, painter :QtGui.QPainter, rect :QtCore.QRectF,
		             initKwargs:T.Optional[T.Dict[str, T.Any]]=None) -> None:
			"""draw icon within rect"""
			painter.drawLine(0.5, 0, 0.5, 1)
			painter.drawLine(0, 0.5, 1, 0.5)

	T = _Base # pycharm doesn't run classmethods in type hints for some reason

class GraphicsMouseSignalMixin:
	"""
	thin interface to add python signals for mouse hover events
	event signature: (self, event:PySide2.QtWidgets.QGraphicsSceneMouseEvent) -> None
	signals are costly to create, so only make them on request

	"""

	def __init__(self):
		self._mousePressed : Signal = None
		self._mouseReleased : Signal = None
		self._mouseMoved : Signal = None
		self._mouseDragged : Signal = None
		self._mouseEntered : Signal = None
		self._mouseLeft : Signal = None

		self.setAcceptHoverEvents(True)

	def mousePressedSignal(self)->Signal:
		if self._mousePressed is None:
			self._mousePressed = Signal("mousePressed")
		return self._mousePressed
	def mouseReleasedSignal(self)->Signal:
		if self._mouseReleased is None:
			self._mouseReleased = Signal("mouseReleased")
		return self._mouseReleased
	def mouseMovedSignal(self)->Signal:
		if self._mouseMoved is None:
			self._mouseMoved = Signal("mouseMoved")
		return self._mouseMoved
	def mouseDraggedSignal(self)->Signal:
		if self._mouseDragged is None:
			self._mouseDragged = Signal("mouseDragged")
		return self._mouseDragged
	def mouseEnteredSignal(self)->Signal:
		if self._mouseEntered is None:
			self._mouseEntered = Signal("mouseEntered")
		return self._mouseEntered
	def mouseLeftSignal(self)->Signal:
		if self._mouseLeft is None:
			self._mouseLeft = Signal("mouseLeft")
		return self._mouseLeft

	def mousePressEvent(self, event:QtWidgets.QGraphicsSceneMouseEvent) -> None:
		self.mousePressedSignal().emit(self, event)
	def mouseReleaseEvent(self, event:QtWidgets.QGraphicsSceneMouseEvent) -> None:
		self.mouseReleasedSignal().emit(self, event)
	def mouseMoveEvent(self, event:QtWidgets.QGraphicsSceneMouseEvent) -> None:
		self.mouseMovedSignal().emit(self, event)
	def hoverEnterEvent(self, event:QtGui.QHoverEvent):
		self.mouseEnteredSignal().emit(self, event)
	def hoverLeaveEvent(self, event:QtGui.QHoverEvent):
		self.mouseLeftSignal().emit(self, event)
