
from __future__ import annotations
import typing as T


from PySide2 import QtWidgets, QtCore, QtGui



class CanvasCamera:
	"""camera for a canvas viewport -
	mainly provides a nicer interface for the camera matrix,
	as well as basic animation

	distributing calls across different views probably
	means doing loops across all calls - maybe that's excessive
	"""

	def __init__(self,
	             #views:T.Iterable[QtWidgets.QGraphicsView]=()
	             ):
		"""init the camera"""
		self.views : list[QtWidgets.QGraphicsView]= []

	def addView(self, view:QtWidgets.QGraphicsView):
		"""add a view to the camera"""
		self.views.append(view)

	def setPosition(self, pos:QtCore.QPointF):
		"""set the position of the camera"""
		for view in self.views:
			view.centerOn(pos)

