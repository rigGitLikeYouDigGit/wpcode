from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtWidgets, QtGui


class ScriptWidget(QtWidgets.QWidget):
	"""live python console widget to execute scripts
	within program

	- autocompletion / inspection with jedi / anakin
	- link / live link to a file on disk
	- diff mode to show changes on top of incoming data
	- line numbers
	- named tabs
	"""

def drawLineNumbers(t:QtWidgets.QTextEdit):
	"""I did a similar effect like 5 years ago at framestore
	"""

	pass

class LineNumberWidget(QtWidgets.QWidget):
	"""
	TODO:
		- mouse tracking, hovering over line numbers
		- click to select whole line
		- drag to move line selection up and down?
	"""
	if T.TYPE_CHECKING:
		def parent(self)->QtWidgets.QTextEdit:pass
	def __init__(self, parent:QtWidgets.QTextEdit,
	             font=QtGui.QFont("arial", pointSize=6),
	             backColour=QtGui.QColor.fromRgbF(0, 0, 0, 0),
	             textColour=QtGui.QColor.fromRgbF(0.3, 0.3, 0.3),
	             hoverColour=QtGui.QColor.fromRgbF(0.6, 0.6, 0.6),
	             selectColour=QtGui.QColor.fromRgbF(0.3, 0.3, 1.0),
	             ):
		super().__init__(parent)

		self.backColour = backColour
		self.textColour = textColour
		self.hoverColour = hoverColour
		self.selectColour = selectColour

		self.lineHeight = self.parent().fontMetrics().height()
		self.setFont(font)
		self.lineRects = []
		self.maxRectWidth = 0
		self.mousePos = QtCore.QPoint()
		self.updateLineRects()

		self.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
		self.setMouseTracking(True)

	def mouseMoveEvent(self, event):
		self.mousePos = event.pos()
		super().mouseMoveEvent(event)

	def leaveEvent(self, event):
		"""clear mouse position"""
		self.mousePos = QtCore.QPoint()

	def updateLineRects(self):
		self.lineRects = self.getLineRects()
		self.maxRectWidth = max(i.width() for i in self.lineRects)
		self.repaint()

	def getLineRects(self)->list[QtCore.QRect]:
		"""return varying widths of rect depending on
		digit count - conform these afterwards if desired
		TODO: could gather a lot more info here, maybe worth having
			a general function to get "line info", like bounding rect,
			how many lines wrapped, etc
		"""
		rects = []
		realLineCount = 0
		lineCount = 0
		ownMetrics = self.fontMetrics()
		parentMetrics = self.parent().fontMetrics()
		wrapLimit = self.parent().rect().width()
		for line in self.parent().toPlainText().split("\n"):
			advance = parentMetrics.horizontalAdvance(line)
			lineCount += 1
			wrapCount = int(advance) // int(wrapLimit) + 1
			toWrite = str(lineCount)
			rectWidth = ownMetrics.horizontalAdvance(toWrite)
			rectHeight = self.lineHeight * wrapCount
			rects.append(QtCore.QRect(0, self.lineHeight * realLineCount,
			                          rectWidth, rectHeight))
			realLineCount += wrapCount
		return rects


	def paintEvent(self, event:QtGui.QPaintEvent):
		"""split by newlines, then modulo each with parent's
		wrap length to get empty lines"""
		painter = QtGui.QPainter(self)
		mousePos = self.mousePos
		if self.mousePos == QtCore.QPoint(): # if mouse is off widget, don't draw highlight
			mousePos = QtCore.QPoint(-100, -100)
		for lineN, rect in enumerate(self.lineRects):
			colour = self.backColour
			if rect.contains(mousePos):
				colour = self.hoverColour
			painter.fillRect(rect, colour)
			painter.drawText(rect, str(lineN + 1),
			                 QtGui.QTextOption(QtCore.Qt.AlignLeft))


class ScriptEditWidget(QtWidgets.QTextEdit):
	"""main meat of autocompletion"""

	def __init__(self, parent=None):
		super().__init__(parent)
		self.lineNumberW = LineNumberWidget(parent=self)
		self.setTabChangesFocus(False)
		self.setLineWrapMode(self.LineWrapMode.WidgetWidth)

	def resizeEvent(self, e):
		super().resizeEvent(e)
		self.setContentsMargins(self.lineNumberW.maxRectWidth, 0, 0, 0)
		self.lineNumberW.setGeometry(
			0, 0, self.lineNumberW.maxRectWidth, self.height())

