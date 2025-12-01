from __future__ import annotations
import types, typing as T
import pprint

import numpy as np

from wplib import log
import copy
from dataclasses import dataclass, asdict, fields, field

from PySide6 import QtCore, QtWidgets, QtGui

from wpui.canvas import WpCanvasElement, WpCanvasScene, WpCanvasView

from wpm import cmds, om, oma, WN, Plug

"""
custom graphics node:
- remembers its position per named graph view
- displays attributes in order added by its bases
	- filters attributes against config of 'interesting' attrs
"""

def defaultField(obj):
	return field(default_factory=lambda: copy.copy(obj))

@dataclass
class ViewNodeData:
	pos : np.ndarray
	expanded : list[int] = defaultField([])


NAME_MODE_NODE_NAME = 0
NAME_MODE_TYPE_NAME = 1
NAME_MODE_NONE = 2

@dataclass
class ViewerData:
	name : str
	pos: np.ndarray = np.array([0.0, 0.0])
	zoom : float = 1.0
	nodeNameMode : int = NAME_MODE_NODE_NAME
	#expanded : list[str] = defaultField([])


class QNodePlug(QtWidgets.QGraphicsEllipseItem):
	
	def __init__(self, plug:Plug, parent=None,
	             isOutput=False
	             ):
		super().__init__(parent)
		self.setRect(0, 0, 10, 10)
		self.plug = plug
		self.isOutput = isOutput
		

class QNodePlugRow(QtWidgets.QGraphicsRectItem):
	"""all rows are direct parents of node, no matter
	tree structure of plugs?
	"""
	def __init__(self, parent:QNodeItem,
	             name: str,
	             plug: Plug=None,
	             inputPlug:QNodePlug=None,
	             outputPlug:QNodePlug=None,
	             parentRow:QNodePlugRow=None,
	             expandBtn:str="",
	             depth=0
	             ):
		super().__init__(parent)
		self.name = name
		self.plug = plug
		self.inputPlug = inputPlug
		self.outputPlug = outputPlug
		self.parentRow = parentRow
		self.expandBtn = expandBtn
		self.depth = depth

		if self.inputPlug:
			self.inputPlug.setParentItem(self)
		if self.outputPlug:
			self.outputPlug.setParentItem(self)

		self.text = QtWidgets.QGraphicsTextItem(
			"  " * depth + self.name, self
		                                        )
		self.setRect(self.text.boundingRect())
		
	def setRect(self, rect, /):
		"""override to keep plug objects following along"""
		super().setRect(rect)
		self.inputPlug.setPos(
			-self.inputPlug.rect().width(),
			0,
		)
		self.inputPlug.setPos(
			-self.rect().width(),
			0,
		)


class QNodeItem(
	QtWidgets.QGraphicsRectItem,
	WpCanvasElement,
):

	def __init__(self, node:om.MObject, data:ViewNodeData, parent=None):
		QtWidgets.QGraphicsRectItem.__init__(self, parent)
		# no core object for now
		WpCanvasElement.__init__(self, node)
		self.wn = WN(self.obj)
		self.data = data

		# group plugs by node parent class
		self.plugSections = []
		self.rows : list[QNodePlugRow] = []

		self.nameLine = QtWidgets.QGraphicsTextItem(self.wn.name(), self)

		self.build()

	def buildRows(self,
	              parentRow:QNodePlugRow,
	              wPlug:Plug,
	              expandedSet:set[str],
	              depth=0
	              ):
		mfnAttr = om.MFnAttribute(wPlug.attribute())
		inPlugItem = None
		if mfnAttr.writable:
			inPlugItem = QNodePlug(wPlug, isOutput=False)
		outPlugItem = None
		if mfnAttr.readable:
			outPlugItem = QNodePlug(wPlug, isOutput=True)

		expandBtn = ""
		if wPlug.branches:
			expandBtn = "+"

		row = QNodePlugRow(
			parent=parentRow,
			name=str(wPlug.name),
			plug=wPlug,
			inputPlug=inPlugItem,
			outputPlug=outPlugItem,
			expandBtn=expandBtn,
			depth=depth
		)
		for i in wPlug.branches:
			self.buildRows(row, i, expandedSet, depth + 1)


	def build(self):
		"""rebuild all items from scratch -
		run over attributes, track longest names,
		"""
		y = 0
		maxLen = 0

		for i in self.rows:
			self.scene().removeItem(i)
		self.rows.clear()

		expanded = {}

		parentRow = self
		depth = 0
		for i, baseCls in enumerate(type(self.wn).__mro__):
			baseCls : type[WN]
			if baseCls is WN:
				break
			toAdd = baseCls.nodeLeafPlugs
			if not toAdd:
				continue
			if i: # set up parent row for attributes
				newRow = QNodePlugRow(
						self,
						baseCls.__name__,
						None, None,
						None,
						"+"
					)
				self.rows.append(
					newRow
				)
				parentRow = newRow

			for attrName in toAdd:
				wPlug = self.wn.plug(attrName)
				self.buildRows(parentRow,
				               wPlug,
				               set(),
				               depth)














