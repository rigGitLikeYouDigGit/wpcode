
from __future__ import annotations

import math
import typing as T

import numpy as np

from PySide2 import QtWidgets, QtCore, QtGui
#from param import rx

from wplib import log, sequence
from wptree import Tree
from wpdex import WpDexProxy
from wplib.serial import Serialisable

from wpui.keystate import KeyState
from wpui import lib as uilib

"""
bases for items in canvas, 
functions for selecting, dragging, etc
"""


class ModelBased(Serialisable):
	"""defining objects that draw from an internal primitive
	data model
	TODO: - obviously move this
	 - don't bother rolling this back into tree etc yet, til
	    we know it's useful
	    """

	@classmethod
	def newModel(cls):
		raise NotImplementedError

	def __init__(self, model=None):
		self.model = model or self.newModel()

	# region serialisation -
	# literally just serialise and store the model
	def encode(self, encodeParams:dict=None) ->dict:
		"""a hierarchy of model objects will only be serialised once,
		at the root"""
		return self.model

	@classmethod
	def decode(cls, serialData:dict, decodeParams:dict=None) ->T.Any:
		return cls(serialData)




class CanvasField(Tree):
	pass


def newSceneModel():
	t = CanvasField("canvas")
	t["items"] = {} # should be a list of typed item datas


class MoveableGraphicsItem(QtWidgets.QGraphicsItem):
	"""test base class for things that can be moved and dragged"""








base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QGraphicsItem

class WpCanvasItem(base,
                   # Serialisable
                   ):
	"""
	base class for moveable, selectable items in canvas
	for ease, we don't disagree with the normal selection system


	how would we set the positions of qt items reactively?

	TODO: allow in world-space, local space, or view space for ui elements
		for view, allow pinning to edges, etc, setting offset from edges, corners
	"""

	@classmethod
	def newDataModel(cls):
		return {"name" : "",
		        "position" : [0, 0]}

	if T.TYPE_CHECKING:
		#parent:
		pass

	def __init__(self,
	             parent:QtWidgets.QGraphicsItem=None,
	             data:WpDexProxy=None,
	             selectable=True,
	             movable=True,
	             ):
		super().__init__(parent=parent)
		#data.ref("pos").drive( self.setPos)
		self._selectable = selectable
		self._movable = movable

	def isSelectable(self)->bool:
		return self._selectable
	def setSelectable(self, state):
		self._selectable = state

	def isMoveable(self)->bool:
		return self._movable
	def setMovable(self, state):
		self._movable = state

	def paint(self, painter, option, widget=...):
		QtWidgets.QGraphicsRectItem.paint(self, painter, option, widget)


if __name__ == '__main__':

	app = QtWidgets.QApplication()

	scene = WpCanvasScene()
	w = WpCanvasView(parent=None, scene=scene,
	                 )
	# item = WpCanvasItem(parent=None,
	#                     )
	item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, 0, 100, 100))
	item.setBrush(QtGui.QBrush(QtCore.Qt.red))
	item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
	scene.addItem(item)
	scene.centreItem(item)
	w.show()
	app.exec_()
