


from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.object import Adaptor, VisitAdaptor

from wpui.model import iterAllItems

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
	pass


class SuperModel(QtGui.QStandardItemModel, Adaptor):
	"""not the photographic kind, this is way better

	supermodel tracks the items needed to mirror each layer of a python structure
	a model for a primitive will have a single item, a model for a list will have
	one item for each entry in the list, and so on

	everything constructed on init

	model does not check if child objects needed, superitem controls that

	put as little behaviour here as needed, only if we need close
	interaction with view widgets - owner superitem should handle
	most
	"""

	adaptorTypeMap = Adaptor.makeNewTypeMap()
	#forTypes = (object, )

	wpDataChanged = QtCore.Signal(dict) # {"item" : QtGui.QStandardItem}

	def __init__(self,
	             wpPyObj: T.Any,
	             wpChildType : VisitAdaptor.ChildType.T() = None,
	             parent: T.Optional[QtCore.QObject] = None,
	             ):
		super(SuperModel, self).__init__(parent=parent)
		self.wpChildType = wpChildType
		self.wpPyObj = wpPyObj



class SuperItem(QtGui.QStandardItem, Adaptor):
	"""inheriting from standardItem for the master object,
	since that lets us reuse the model/item link
	to map out structure.

	SuperItem for a given type should gather together
	all associated types - model, view and delegate.

	SuperItem may also reach through its child model to create
	new SuperItems directly - easier to follow than splitting
	behaviour across classes
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
		self.wpVisitAdaptor = VisitAdaptor.adaptorForObject(pyObj)


		# check that item has a visitor class
		assert self.wpVisitAdaptor, f"no VisitAdaptor for {self.wpPyObj, type(self.wpPyObj)}"

		if self._needsModel():
			# build model
			newModelType = SuperModel.adaptorForObject(pyObj)
			assert newModelType, f"no SuperModel adaptor type for {pyObj, type(pyObj)}"
			self.wpItemModel = newModelType(pyObj,
			                        wpChildType=self.wpChildType,
			                        parent=self.wpParentQObj)
			# build stuff
			itemChildTypeList: list[
				tuple[QtGui.QStandardItem, VisitAdaptor.ChildType.T()]] = self._generateItemsFromPyObj()
			self._insertItemsToModel(itemChildTypeList)

	def _generateItemsFromPyObj(self) -> list[tuple[SuperItem, VisitAdaptor.ChildType.T()]]:
		""" OVERRIDE if you want
		generate items from python object,
		add them to the model

		reuse visitor here - no obligation to use
		all child objects if they make no sense for UI

		objects are initialised and parent connections made before
		items are added to model - bit weird
		"""
		resultItems = []
		childObjs: VisitAdaptor.ITEM_CHILD_LIST_T = self.wpVisitAdaptor.childObjects(self.wpPyObj)
		for obj, childType in childObjs:
			childItemType = SuperItem.adaptorForObject(obj)
			assert childItemType, f"no SuperItem adaptor type for {obj, type(obj)}"

			# this will initialise and connect child models too
			item = childItemType(obj, wpChildType=childType, parentQObj=self)
			resultItems.append((item, childType))
		return resultItems

	def wpChildSuperItems(self)->list[SuperItem]:
		"""return a list of all child items"""
		return list(filter(lambda x: isinstance(x, SuperItem), iterAllItems(self.wpItemModel)))

	def _insertItemsToModel(self,
	                        items: list[tuple[QtGui.QStandardItem,
	                        VisitAdaptor.ChildType.T()]]):
		"""
		OVERRIDE for things like maps, trees etc
		insert items into the model"""
		for item, childType in items:
			self.wpItemModel.appendRow(item)


	def _needsModel(self)->bool:
		"""return True if this item needs a model -
		False for strings, ints, primitive types
		True for containers"""
		return bool(VisitAdaptor.adaptorForObject(self.wpPyObj).childObjects(self.wpPyObj))


	@classmethod
	def forData(cls, data:T.Any,
	            wpChildType : VisitAdaptor.ChildType.T() = None,
	            parentQObj: QtCore.QObject = None
	            )->SuperItem:
		"""return a SuperItem structure for the given data"""
		itemType : type[SuperItem] = cls.adaptorForObject(data)
		assert itemType, f"no SuperItem adaptor type for {data, type(data)}"
		return itemType(data)






