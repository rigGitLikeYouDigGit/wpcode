


from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from dataclasses import dataclass

from wplib import log
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

def defaultGetModelClsFn(forObj:T.Any)->T.Type[SuperModel]:
	"""return default model class for given object"""
	return SuperModel.adaptorForObject(forObj)

def defaultGetItemClsFn(forObj:T.Any)->T.Type[SuperItem]:
	"""return default item class for given object"""
	return SuperItem.adaptorForObject(forObj)

@dataclass
class SuperModelParams:
	"""parametre class for model and/or ui behaviour
	override getModelClsFn and getItemClsFn to
	inject dependency when generating new item or model classes for
	py object types
	"""
	getModelClsFn : T.Callable[[T.Any], T.Type[SuperModel]] = defaultGetModelClsFn
	getItemClsFn : T.Callable[[T.Any], T.Type[SuperItem]] = defaultGetItemClsFn



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
	             parentSuperItem: T.Optional[SuperItem] = None,
	             parent: T.Optional[QtCore.QObject] = None,
	             ):
		super(SuperModel, self).__init__(parent=parent)
		self.wpChildType = wpChildType
		self.wpPyObj = wpPyObj
		self.wpParentSuperItem : SuperItem = parentSuperItem



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
	             parentSuperItem: T.Optional[SuperItem] = None
	             ):
		super(SuperItem, self).__init__()
		self.wpPyObj = pyObj
		self.wpChildType = wpChildType
		self.wpItemModel : SuperModel = None
		self.wpParentQObj = parentQObj
		#self.wpVisitAdaptor = VisitAdaptor.adaptorForObject(pyObj)
		self.wpVisitAdaptor = self._getComponentTypeForObject(pyObj, component="visitor")
		self.wpParentSuperItem : SuperItem = None


		# check that item has a visitor class
		assert self.wpVisitAdaptor, f"no VisitAdaptor for {self.wpPyObj, type(self.wpPyObj)}"

		if self._needsModel():
			# build model
			# newModelType = SuperModel.adaptorForObject(pyObj)
			newModelType = self._getComponentTypeForObject(pyObj, component="model")
			assert newModelType, f"no SuperModel adaptor type for {pyObj, type(pyObj)}"
			if self.wpParentQObj:
				#log("PARENT", self.wpParentQObj, type(self.wpParentQObj))
				assert isinstance(self.wpParentQObj, QtCore.QObject)
			self.wpItemModel = newModelType(pyObj,
			                        wpChildType=self.wpChildType,
			                        parent=self.wpParentQObj,
			                                parentSuperItem=self)
			# build stuff
			#log("GEN ITEMS FOR ", self.wpPyObj, self.wpChildType, self.wpVisitAdaptor)
			itemChildTypeList: list[
				tuple[QtGui.QStandardItem, VisitAdaptor.ChildType.T()]] = self._generateItemsFromPyObj()
			self._insertItemsToModel(itemChildTypeList)

	def _getComponentTypeForObject(self, forObj, component="model"):
		"""return the type of object used for the given part of the system -
		use this to inject dependency with params struct later.

		By default we rely on adaptor lookup, which is limited to 1:1 relations
		"""
		if component == "model":
			return SuperModel.adaptorForObject(forObj)
		elif component == "item":
			return SuperItem.adaptorForObject(forObj)
		elif component == "visitor":
			return VisitAdaptor.adaptorForObject(forObj)
		else:
			raise ValueError(f"unknown component type {component}")


	def _generateItemsFromPyObj(self) -> list[tuple[SuperItem, VisitAdaptor.ChildType.T()]]:
		""" OVERRIDE if you want
		generate items from python object,
		add them to the model

		reuse visitor here - no obligation to use
		all child objects if they make no sense for UI

		objects are initialised and parent connections made before
		items are added to model - bit weird

		we set childtype on object, and return it in tuples - might
		be redundant
		"""
		resultItems = []
		childObjs: VisitAdaptor.ITEM_CHILD_LIST_T = self.wpVisitAdaptor.childObjects(self.wpPyObj)
		for obj, childType in childObjs:
			#childItemType = SuperItem.adaptorForObject(obj)
			childItemType = self._getComponentTypeForObject(obj, component="item")
			assert childItemType, f"no SuperItem adaptor type for {obj, type(obj)}"
			#log("GEN ITEM", obj, childType, childItemType, self.wpItemModel, self)

			# this will initialise and connect child models too
			item = childItemType(obj, wpChildType=childType, parentQObj=self.wpItemModel,
			                     parentSuperItem=self)
			resultItems.append((item, childType))
		return resultItems

	def wpChildSuperItems(self)->list[SuperItem]:
		"""return a list of all child items"""
		return list(filter(lambda x: isinstance(x, SuperItem),
		                   iterAllItems(model=self.wpItemModel)))

	def wpResultObj(self)->T.Any:
		"""retrieve new object from this item's child superItems"""
		log("wpResultObj", self, self.wpPyObj, self.wpChildType)
		raise NotImplementedError

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
		return bool(self.wpVisitAdaptor.childObjects(self.wpPyObj))


	@classmethod
	def forData(cls, data:T.Any,
	            wpChildType : VisitAdaptor.ChildType.T() = None,
	            parentQObj: QtCore.QObject = None
	            )->SuperItem:
		"""return a SuperItem structure for the given data"""
		itemType : type[SuperItem] = cls.adaptorForObject(data)
		assert itemType, f"no SuperItem adaptor type for {data, type(data)}"
		return itemType(data)






