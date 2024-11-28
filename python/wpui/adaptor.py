from __future__ import annotations
import typing as T

import numpy as np
from PySide2 import QtCore, QtWidgets, QtGui

from wplib.object import VisitAdaptor
from wpdex import WpDex
from wptree import Tree
from wpui import lib
from wplib.maths import arr, NPArrayLike
from wplib.object import OverrideProvider

class Q1DArrayLike(NPArrayLike):
	"""assume for now that arrays are always in floats for calculation,
	and hope we can treat the F types in qt the same as otherwise"""
	forTypes = (QtCore.QPoint, QtCore.QPointF,
	            QtCore.QSize, QtCore.QSizeF)
	def __array__(self:QtCore.QPoint,
	              dtype=None, copy=None):
		return np.array(self.toTuple(), dtype=dtype)
	@classmethod
	def fromArray(cls, ar, **kwargs):
		return cls(*ar)
class QColorArrayLike(NPArrayLike):
	"""no real reason to keep this separate, just IN CASE
	we ever find some better way of handling HSV / RGB initialisation
	by default we work with normalised 0-1 colour values

	DO WE ASSUME COLOURS TO BE 4-LONG?
	sure, try it for now, change if it turns out to be annoying"""
	forTypes = (QtGui.QColor, )
	def __array__(self:QtGui.QColor, dtype=None, copy=None):
		return np.array(self.getRgbF())
	@classmethod
	def fromArray(cls:QtGui.QColor, ar, **kwargs):
		return cls.fromRgbF(*ar)
class Q2DArrayLike(NPArrayLike):
	forTypes = (QtCore.QLine, QtCore.QLineF)
	def __array__(self: QtCore.QLine,
	              dtype=None, copy=None):
		return np.array(self.toTuple(), dtype=dtype)

	@classmethod
	def fromArray(cls:type[QtCore.QLine], ar, **kwargs):
		pointType = QtCore.QPoint if cls == QtCore.QLine else QtCore.QPointF
		return cls(pointType(*ar[0]),
		           pointType(*ar[1]))
class QPolygonArrayLike(NPArrayLike):
	forTypes = (QtGui.QPolygon, QtGui.QPolygonF)
	def __array__(self: QtGui.QPolygon,
	              dtype=None, copy=None):
		return np.array(self.toList(), dtype=dtype)
	@classmethod
	def fromArray(cls:type[QtGui.QPolygon], ar, **kwargs):
		pointType = QtCore.QPoint if cls == QtGui.QPolygon else QtCore.QPointF
		return cls.fromList([pointType(*i) for i in ar])

# visitors
class WidgetVisitAdaptor(VisitAdaptor):
	forTypes = (QtWidgets.QWidget, )
	@classmethod
	def childObjects(cls, obj:T.Any, params:PARAMS_T) ->CHILD_LIST_T:
		"""return only widgets that have a name"""
		result = []
		for k, v in lib.widgetChildMap(obj, includeObjects=False).items():
			data = VisitAdaptor.ChildData(
				key=k, obj=v, data={}
			)
			result.append(data)
		return result

class QGraphicsVisitAdaptor(VisitAdaptor):
	forTypes = (QtWidgets.QGraphicsItem, )
	@classmethod
	def childObjects(cls, obj:QtWidgets.QGraphicsItem, params:PARAMS_T) ->CHILD_LIST_T:
		"""return indexed map of child graphicsItem object"""
		return [VisitAdaptor.ChildData(
			key=i, obj=v, data={}
		) for i, v in enumerate(obj.childItems())]

def qtOverrideAncestorsFn(obj, *args, **kwargs):
	"""function providing ancestors for the given qt item
	could add in a LOT more complex logic, should a graphics item continue to its scene,
	its view, etc
	"""
	if isinstance(obj, QtCore.QObject):
		if not obj.parent(): return ()
		return (obj.parent(), )
	if isinstance(obj, QtWidgets.QGraphicsItem):
		return obj.parentItem() if obj.parentItem() else obj.scene()


class WidgetDex(WpDex):
	"""allowing traversal through named qwidgets with paths -
	MAYBE extend to any QObjects, but widgets are enough for now"""
	forTypes = (QtWidgets.QWidget, )




