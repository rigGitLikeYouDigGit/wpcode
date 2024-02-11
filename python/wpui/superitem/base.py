


from __future__ import annotations
import pprint, copy, textwrap, ast
import typing as T

import PySide2
from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel
from wplib.object import PluginRegister, PluginBase, Adaptor, VisitAdaptor
from wplib import log

from wpui.model import iterAllItems
from wpui import model as libmodel

from wpui.widget.table import WPTableView, WPTreeView

"""what if whole thing in one

super-system takes care of layout and rearranging - 
should there be something like a super-delegate, to allow custom
drawing of embedded widgets?

consider that the outer superWidget allows defining defaults for
different types, and then further overrides against paths within structure


MODELS MUST BE SHARED
can't have one simple model for each view, as we 
need to capture and reflect changes to primitive and immutable types


at top level, a super-system instance has a signal for dataChanged() - 
that can be connected to whatever outer hook, to update the system as a whole


class structure here may be excessive, but we need custom classes for
model, item, and widget - each of these handled with Adaptors.
Could also reuse a lot of the Visitor stuff, since we're doing the same
operations of traversing and updating a structure.


TEST coding standards for this system - use a "namespace" of "wp" for all
attributes added to QT types -
I don't trust anything anymore
"""

if T.TYPE_CHECKING:
	from wptree import Tree, TreeInterface
	from wptree.ui import TreeSuperItem


class SuperModel(QtGui.QStandardItemModel, Adaptor):
	"""not the photographic kind, this is way better

	supermodel tracks the items needed to mirror each layer of a python structure
	a model for a primitive will have a single item, a model for a list will have
	one item for each entry in the list, and so on

	everything constructed on init
	"""

	adaptorTypeMap = Adaptor.makeNewTypeMap()

	wpDataChanged = QtCore.Signal(dict) # {"item" : QtGui.QStandardItem}

	def __init__(self,
	             wpPyObj: T.Any,
	             wpChildType : VisitAdaptor.ChildType.T() = None,
	             parent: T.Optional[QtCore.QObject] = None,
	             ):
		super(SuperModel, self).__init__(parent=parent)
		self.wpChildType = wpChildType
		self.wpPyObj = wpPyObj

		# build stuff
		itemChildTypeList : list[tuple[QtGui.QStandardItem, VisitAdaptor.ChildType.T()]] = self._generateItemsFromPyObj()

	def _generateItemsFromPyObj(self):
		""" OVERRIDE if you want
		generate items from python object,
		add them to the model

		reuse visitor here - no obligation to use
		all child objects if they make no sense for UI
		"""
		childObjs : VisitAdaptor.ITEM_CHILD_LIST_T = (
			VisitAdaptor.adaptorForObject(self.wpPyObj).childObjects(
				self.wpPyObj))
		for obj, childType in childObjs:
			childItemType = SuperItem.adaptorForObject(obj)
			assert childItemType, f"no SuperItem adaptor type for {obj}"




	def _insertItemsToModel(self,
	                        items: list[tuple[QtGui.QStandardItem,
	                        VisitAdaptor.ChildType.T()]]):
		"""insert items into the model"""
		pass

	def resultObj(self):
		"""return the result object
		from child items"""
		raise NotImplementedError

class SuperItem(QtGui.QStandardItem, Adaptor):
	"""inheriting from standardItem for the master object,
	since that lets us reuse the model/item link
	to map out structure.

	SuperItem for a given type should gather together
	all associated types - model, view and delegate
	"""

	adaptorTypeMap = Adaptor.makeNewTypeMap()

	def __init__(self,
	             pyObj: T.Any,
	             wpChildType : VisitAdaptor.ChildType.T() = None,
	             parentQObj: QtCore.QObject = None,
	             ):
		super(SuperItem, self).__init__()
		self.wpPyObj = pyObj
		self.wpChildType = wpChildType
		self.wpItemModel : SuperModel = None
		self.wpParentQObj = parentQObj

		if self._needsModel():
			self.wpItemModel = self._makeChildModelForPyObj(pyObj)

	def _needsModel(self)->bool:
		"""return True if this item needs a model -
		False for strings, ints, primitive types
		True for containers"""
		return bool(VisitAdaptor.adaptorForObject(self.wpPyObj).childObjects(self.wpPyObj))

	def _makeChildModelForPyObj(self, pyObj: T.Any):
		"""make child model, set its parent
		call to Adaptor may be excessive here,
		but theoretically you could make items and
		models that map to different types compatible
		"""
		newModelType = SuperModel.adaptorForObject(pyObj)
		assert newModelType, f"no SuperModel adaptor type for {pyObj}"
		newModel = newModelType(pyObj,
		                        wpChildType=self.wpChildType,
		                        parent=self.wpParentQObj)
		return newModel



class ListSuperModel(SuperModel):
	"""model for a list"""

	forTypes = (list,)

	def _generateItemsFromPyObj(self):
		"""generate items from python object"""
		for i in self.pyObj:
			pluginItem = self.forValue(i)
			self.appendRow(
				pluginItem
			)

if __name__ == '__main__':
	import sys
	import qt_material
	app = QtWidgets.QApplication(sys.argv)
	qt_material.apply_stylesheet(app, theme='dark_blue.xml')

	model = SuperModel([1, 2, 3])


	sys.exit(app.exec_())




