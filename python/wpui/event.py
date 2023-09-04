from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets


class WheelEventFilter(QtCore.QObject):
	def eventFilter(self, obj, event):
		if event.type() == QtCore.QEvent.Wheel:
			print("ate wheel event")
			return True
		else:
			return QtCore.QObject.eventFilter(self, obj, event)

class AllEventEater(QtCore.QObject):
	# the hunger
	def eventFilter(self, watched:QtCore.QObject,
					event:QtCore.QEvent):
		if not any([event.type() == i for i in [
			QtCore.QEvent.MouseMove,
			QtCore.QEvent.MouseButtonPress,
			QtCore.QEvent.MouseButtonDblClick,
			QtCore.QEvent.GraphicsSceneContextMenu,
			QtCore.QEvent.GraphicsSceneDragMove,
			QtCore.QEvent.KeyPress
		]]):
			return QtCore.QObject.eventFilter(self, watched, event)
		#print("stopped", event)
		return True