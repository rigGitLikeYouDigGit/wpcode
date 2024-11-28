from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtWidgets, QtGui
from wptree import Tree

from wplib import inheritance, nxlib
from wplib.maths import arr,fromArr
from wpdex.ui.atomic import ExpWidget
from wpdex import WpDexProxy, WX, react
from wpdex.ui import AtomicUiInterface
from wpui.widget.collapsible import ShrinkWrapWidget
from .element import WpCanvasElement

if T.TYPE_CHECKING:
	from .node import NodeDelegate
	from .element import WpCanvasElement
	from .scene import WpCanvasScene

"""
how the heck do we handle drawing connection points within scene

each "end" of a connection has to update all its related items on movement

all connection points will be CanvasElement objects 

"""
# base = object
# if T.TYPE_CHECKING:
# 	base = WpCanvasElement
class ConnectionPoint(WpCanvasElement):
	"""this shouldn't know much
	for now we assume a point is either a source or destination,
	as opposed to assuming connections are always dragged from
	source to destination

	HOVER STATES for mouse -
	on mouse nearby, show circle over the connection point -


	"""

	if T.TYPE_CHECKING:
		def scene(self)->WpCanvasScene: pass
	def __init__(self, isInput=True,
	             hoverRange=30):
		self.isInput = isInput
		self.hoverRange = hoverRange
		self.isHovered = False
		self.setAcceptHoverEvents(True)
		self.setAcceptDrops(True)



	def connectionLines(self)->list[ConnectionGroupDelegate]:
		"""return all connection line delegates attached to this point"""
		return list(self.scene().connectedItems(self, key="connectGroup"))

	def paint(self, *args, **kwargs):
		"""if mouse is nearby draw a template connection to show
		drag can be done

		if drag is in progress, check through all connections
		and grey out all those that cannot accept it
		"""

	def connectionPoint(self,
	                    forDelegate:ConnectionGroupDelegate=None
	                    )->tuple[tuple[float, float], (tuple[float, float], None)]:
		"""implement custom logic for the given delegate if wanted
		by default falls back to nearest point on outline, on bounding
		rect, etc

		optionally also return a tuple for the direction vector to use for the connection

		"""

	def connectionPath(self, forDelegate:ConnectionGroupDelegate=None)->QtGui.QPainterPath:
		"""in case a connection can be made / slide along a patch on the given
		object
		by default return None"""

	#region drag processing
	def canAcceptDragConnections(self):
		"""overall method - if False, this won't be included in scene event checks
		for valid targets when dragging a connection"""
		return False

	def canCreateDragConnections(self):
		return True

	def acceptsIncomingConnection(self, fromObj:ConnectionPoint)->bool:
		"""check if this point accepts a drag connection from the given source"""
		return False

	def acceptsOutgoingConnection(self, toObj:ConnectionPoint)->bool:
		"""check if a line from this point can attach to the given point
		both of the above must agree for a connection to be made"""
		return False

	def onIncomingConnectionAccepted(self, fromObj:ConnectionPoint)->bool:
		pass
	def onOutgoingConnectionAccepted(self, toObj:ConnectionPoint)->bool:
		pass

	def hoverMoveEvent(self, event):
		self.update()
	def hoverEnterEvent(self, event):
		self.update()
	def hoverLeaveEvent(self, event):
		self.update()

	def mousePressEvent(self, event):
		if self.isUnderMouse():
			self.scene().onConnectionDragBegin(fromObj=self)
		return super().mousePressEvent(event)


	# def dragLeaveEvent(self, event):
	# 	self.scene().onConnectionDragBegin(fromObj=self)
	#endregion

class ConnectionGroupDelegate(WpCanvasElement):
	"""
	holds start and end objects that could be ANYTHING - graphicsItems, widgets, points, etc

	need some way to flag when they move, or just sync at all times -

	for each end, first check if it defines a "connectionPoint()" method, and if so, check if that method returns None for this delegate (somehow, I know this splits the logic around)
	if not, try and check if the end points have a representation in scene - eg like bounding geometry
	if so take the connection points as the mutual closest points there, like for the group fields

	mixin parent class for now, a t some point I might have to rewrite all of this into
	composition somehow
	for now we assume start and end

	TODO: in the long future we could try and make this work with any python objects,
		as long as those objects have a coherent drawing representation in scene or UI
	"""

	if T.TYPE_CHECKING:
		def scene(self)->WpCanvasScene: pass

	def __init__(self,
	             start:ConnectionPoint,
	             end:ConnectionPoint,
	             **kwargs
	             ):
		"""allow static pointers or functions to look up based on
		other data contained in mixin"""
		self.start = start
		self.end = end

	def getConnectionPoints(self)->list[tuple[float, float], tuple[float, float]]:
		"""get the points and vectors to use to draw this delegate"""
		points = [None, None]
		vectors = [None, None]

		if not self.start:
			log("no start point found for delegate", self, self.start, self.end)
		if not self.end:
			log("no end point found for delegate", self, self.start, self.end)
		for i, obj in enumerate((self.start, self.end)):
			if isinstance(obj, ConnectionPoint):
				result = obj.connectionPath(forDelegate=self)
				if result is None:
					result = obj.connectionPoint(forDelegate=self)
				assert result is not None, "Must implement either connectionPath() or connectionPoint()"
				point, vector = result
				if vector is None:
					vector = (0.0, -1.0 if i else 1.0)
				points[i] = point
				vectors[i] = vector
			else:
				raise NotImplementedError("not supported yet:", obj)
		return points, vectors



	def onMouseHover(self,
	                 path:QtWidgets.QGraphicsPathItem,
	                 pos):
		"""called by drawConnections whenever the path associated with this
		element is moused over"""


class PathEdge(QtWidgets.QGraphicsPathItem):

	def __init__(self,
	             ptA:ConnectionPoint,
	             ptB:ConnectionPoint
	             ):
		"""simple approach instead of the separate object for drawing

		TODO: maybe query the painter object to update all paths at once, later
		"""
		self.ptA = ptA
		self.ptB = ptB

	def path(self)->QtGui.QPainterPath:
		"""duplicated from above to get pos and normal vectors from node
		"""
		points = [None, None]
		vectors = [None, None]

		if not self.start:
			log("no start point found for delegate", self, self.start, self.end)
		if not self.end:
			log("no end point found for delegate", self, self.start, self.end)
		for i, obj in enumerate((self.start, self.end)):
			if isinstance(obj, ConnectionPoint):
				result = obj.connectionPath(forDelegate=self)
				if result is None:
					result = obj.connectionPoint(forDelegate=self)
				assert result is not None, "Must implement either connectionPath() or connectionPoint()"
				point, vector = result
				if vector is None:
					vector = (0.0, -1.0 if i else 1.0)
				points[i] = point
				vectors[i] = vector
			else:
				raise NotImplementedError("not supported yet:", obj)
		return points, vectors


class ConnectionsPainter:
	"""no idea -
	it seems that usually you need to draw multiple connections
	in awareness of the whole, not just one by one.

	this way we can have multiple ways to draw connections, without modifying the logic
	of where and how they connect
	"""

	def __init__(self, connections:list[ConnectionGroupDelegate],
	             scene:QtWidgets.QGraphicsScene):
		self.connections = connections

	def draw(self):
		"""build collection of pathItems for each delegate after working out
		paths?
		then track which is matched to which?
		seems super complicated

		- just do setPath(), don't need to put mouse-over and interaction
		logic here
		"""

	def onMouseHover(self):
		"""work out which path is hovered over,
		then which delegate is associated with that path,
		then trigger that delegate's hover method?"""
		# NO



