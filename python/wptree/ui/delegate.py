
from __future__ import annotations

import typing
from PySide2 import QtWidgets, QtGui, QtCore

from wptree.main import Tree

if typing.TYPE_CHECKING:
	from wptree.ui.view import TreeView
	pass

from wptree.ui.constant import treeObjRole, rowHeight


class EasyDelegate(QtWidgets.QStyledItemDelegate

                   ):
	"""make working with delegates slightly less painful
	than pulling teeth

	possible that this is a bit too split out,
	we could reasonably just have methods to return pens and brushes,
	instead of also colours

	"""

	def backgroundColour(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex)->typing.Union[QtGui.QColor, None]:
		"""return a Qt.Color or None"""
		return None

	def backgroundBrush(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex)->typing.Union[QtGui.QBrush, None]:
		"""return the background brush to use to paint this entry
		default is to use the colour set by backgroundColour()"""
		colour = self.backgroundColour(painter, option, index)

		return QtGui.QBrush(colour
		                    or  # QtCore.Qt.NoBrush
		                    wpui.superitem.view.base()
		                    )


	def outlineColour(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex)->typing.Union[QtGui.QColor, None]:
		"""return the colour to use to draw the outline of this entry
		default is None"""
		return None

	def outlinePen(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex)->typing.Union[QtGui.QPen, None]:
		"""return the pen to use to draw the outline of this entry
		default is no pen, for cleanliness"""
		return QtGui.QPen(self.outlineColour(painter, option, index)
		                  or QtCore.Qt.NoPen)

	def textColour(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex)->typing.Union[QtGui.QColor, None]:
		"""return a Qt.Color or None
		in the base, check first for an explicit colour in data,
		then in palette

		passing None to QColor gives black, I didn't know that
		"""
		# check for colour set as data
		dataCol = index.data(QtCore.Qt.TextColorRole)
		# fall back to palette coour
		paletteCol : QtGui.QColor = option.palette.text().colour()
		#print("base textColour", dataCol, paletteCol)
		textCol = dataCol or paletteCol
		return textCol

	def textPen(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex)->typing.Union[QtGui.QPen, None]:
		"""return the pen to use to draw text of this entry
		default is normal black"""
		#print("base text colour", self.textColour(painter, option, index))
		return QtGui.QPen(self.textColour(painter, option, index)
		                  or QtCore.Qt.NoPen
		                  )

	def paint(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex) -> None:
		self._painter = painter
		painter.save()
		self.initStyleOption(option, index)

		# construct new style option object with different colurs, then pass that to super
		newOption = QtWidgets.QStyleOptionViewItem(option)

		# draw outline and background
		backgroundBrush = self.backgroundBrush(painter, newOption, index)
		outlinePen = self.outlinePen(painter, newOption, index)
		if backgroundBrush or outlinePen:
			newOption.Alternate = False  # no override from alternate row colours
			newOption.None_ = True # strange glyphs beyond all knowledge

		painter.setBrush(backgroundBrush)
		painter.setPen(outlinePen)
		painter.drawRoundedRect(newOption.rect, 2, 2)

		# # set pen for text
		#painter.setPen(self.textPen(painter, newOption, index))

		# I'm not sure what the interaction is between the painter
		# and the palette style below for text

		# # reset pen for text
		# textColour = self.textColour(painter, newOption, index)
		# if textColour is None:
		# 	textColour = option.palette.text().color()
		# #print("textColour", textColour)
		# newOption.palette.setColor(
		# 	QtGui.QPalette.Text,
		# 	QtGui.QColor(textColour),
		# )
		painter.setPen(self.textPen(painter, newOption, index))
		#painter.setBrush(QtCore.Qt.NoBrush)

		#painter.drawText(newOption.rect, newOption.text)

		super(EasyDelegate, self).paint(painter, newOption, index)

		painter.restore()


class TreeNameDelegate(
	#QtWidgets.QStyledItemDelegate
	EasyDelegate
	):
	""" use for support of locking entries, options on right click etc
	could also be used for drawing proxy information
	"""


	def __init__(self, parent=None):
		super(TreeNameDelegate, self).__init__(parent)
		self.branch = None #type: Tree

	def sizeHint(self, option:QtWidgets.QStyleOptionViewItem,
	             index:QtCore.QModelIndex) -> QtCore.QSize:
		metrics = QtGui.QFontMetrics(option.font)
		length = metrics.size(
			QtCore.Qt.TextSingleLine,
			index.data(QtCore.Qt.DisplayRole)).width() * 1.1 + 5
		return QtCore.QSize(length, rowHeight)

	def paint(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex) -> None:
		"""set the current branch from index
		more complex painting for deltas, references etc
		is handled by view as a post process, was just easier that way
		"""
		self.branch = index.data(treeObjRole)
		super(TreeNameDelegate, self).paint(painter, option, index)

	def textColour(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex) ->typing.Union[QtGui.QColor, None]:
		"""colour delta'd branches white for contrast"""
		# if self.branch.deltaData():
		# 	return QtCore.Qt.white
		return super(TreeNameDelegate, self).textColour(painter, option, index)


class TreeValueDelegate(EasyDelegate):
	""" used to display more complex processing of value items """

	def sizeHint(self, option:QtWidgets.QStyleOptionViewItem,
	             index:QtCore.QModelIndex) -> QtCore.QSize:
		metrics = QtGui.QFontMetrics(option.font)
		length = metrics.size(
			QtCore.Qt.TextSingleLine,
			index.data(QtCore.Qt.DisplayRole)).width() * 1.1
		return QtCore.QSize(length, rowHeight)

	# def displayText(self, value, locale):
	# 	""" format to remove quotes from strings, lists and dicts """
	# 	#print("delegate called for {}".format(value))
	# 	return value

	def textColour(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex) ->typing.Union[QtGui.QColor, None]:
		""" check if item has a value widget """
		view: TreeView = option.widget
		valueWidget = view.indexWidget(index)
		if valueWidget is not None:
			# do not paint if custom widget is used
			return None
		return super(TreeValueDelegate, self).textColour(painter, option, index)






	# def paint(self, painter:QtGui.QPainter, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex) -> None:
	# 	view: TreeView = option.widget
	# 	self.initStyleOption(option, index)
	# 	newOption = QtWidgets.QStyleOptionViewItem(option)
	# 	valueWidget = view.indexWidget(index)
	# 	if valueWidget is not None:
	# 		# do not paint if custom widget is used
	# 		newOption.text = ""
	# 	#print("text", newOption.text)
	# 	return super(TreeValueDelegate, self).paint(painter, newOption, index)



