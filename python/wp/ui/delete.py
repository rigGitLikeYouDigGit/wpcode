from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui


""" deleting widgets has always been a little weird - 
some conditions create self-sustaining references in python
that mess with weakrefs, but still outlast the c++ object.

Here we delete an object and all its children"""

def emptyLayoutTree(layout:QtWidgets.QLayout):
	"""delete all widgets in a layout"""
	for i in range(layout.count()):
		item = layout.takeAt(0)
		if item is None:
			continue
		try:
			emptyLayoutTree(item.layout())
		except AttributeError:
			pass
		try:
			deleteObjectTree(item.widget())
		except AttributeError:
			pass

		# else:
		# 	raise TypeError(f"cannot delete item: {item}")
		#item = layout.takeAt(i)
		#layout.removeItem(item)
		#item.deleteLater()
	layout.deleteLater()

def deleteObjectTree(obj:(QtWidgets.QWidget, QtCore.QObject)):
	"""delete widget, layouts, objects, etc"""
	if obj is None:
		return
	if isinstance(obj, QtWidgets.QWidget):
		if obj.layout() is not None:
			emptyLayoutTree(obj.layout())

	for i in obj.children():
		deleteObjectTree(i)
	obj.setParent(None)
	obj.deleteLater()



