from __future__ import annotations

import traceback
import types

from weakref import WeakValueDictionary

from PySide2 import QtCore, QtWidgets, QtGui

from wpexp.tostr import toStr
from wptree import Tree
from wplib import inheritance
from wplib.serial import serialise
from wplib.object import Adaptor, PostInitMeta

from wpui import model as libmodel
from wpui.treemenu import ContextMenuProvider

from wpdex import *


class Condition:
	"""EXTREMELY verbose, but hopefully this lets us reuse
	rules and logic as much as possible"""
	def __init__(self, *args,
	             validateFn=lambda v : True,
	             correctFn=lambda v : v,
	             **kwargs,
	             ):
		self.args = args
		self.kwargs = kwargs
		self.validateFn = validateFn
		self.correctFn = correctFn
	def validate(self, value, *args, **kwargs):
		"""
		return None -> all fine
		raise error -> validation failed
		return (error, val) -> failed with suggested result
		"""
		return self.validateFn(value, *args, **kwargs)

	def correct(self, value):
		raise NotImplementedError

	@classmethod
	def checkConditions(cls, value, conditions:T.Sequence[Condition],
	                    *args, **kwargs):
		"""TODO: consider some kind of "report" object
		to collect the results of multiple conditions
		at once"""
		for i in conditions:
			# try just passing a raw function
			if isinstance(i, (types.FunctionType, types.MethodType)):
				assert i(value, *args, **kwargs)
				continue
			i.validate(value, *args, **kwargs)


base = (Adaptor, )
if T.TYPE_CHECKING:
	base = (QtWidgets.QWidget, Adaptor)

base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QWidget

class AtomicUiInterface(
	#Adaptor,
	base,
	#inheritance.MetaResolver,
	metaclass=PostInitMeta
):
	"""base class to pull out the common logic of validation
	for both widgets and view items, since either one might be
	more useful

	standardItems aren't QObjects, so we need to route signals
	through the model? and it gets REALLY messy
	"""

	# def __init_subclass__(cls, **kwargs):
	# 	log("atomic base init subclass", cls, kwargs)

	def __init__(self, value,
	             conditions=(),
	             warnLive=False,
	             commitLive=False,
	             enableInteractionOnLocked=False,
	             **kwargs):
		# set up all reactive elements first -
		self._proxy : WpDexProxy = None
		self._dex : WpDex = None
		#self._value : WX = rx(None)
		self._value : WX = WX(None)

		# child widgets of container may not be direct children in Qt
		self._childAtomics : dict[WpDex.pathT, AtomicWidgetOld] = WeakValueDictionary()

		#self._value = rx(value) if not isinstance(value, rx) else value
		self._immediateValue = rx(None) # changes as ui updates


		self.conditions = conditions
		self.warnLive = warnLive

		self.setValue(value)

	def _setErrorState(self, state, exc=None):
		"""NO IDEA how best to integrate this, but should have a flag
		to warn through ui, without trying to commit fully"""

	def _syncImmediateValue(self, *args, **kwargs):
		#log("sync immediate")
		try:
			val = self._processResultFromUi(self._rawUiValue())
		except Exception as e:
			"""if an invalid input occurs as you type, 
			warn user and don't try to commit it"""
			self._setErrorState(True, e)
			return

		#log("processed", val)
		self._immediateValue.rx.value = val


	def _fireDisplayEdited(self, *args):
		"""TODO: find a way to stop double-firing this
			if you connect 2 signals to it
			"""
		self._syncImmediateValue()
		self.displayEdited.emit(args)

	def _fireDisplayCommitted(self, *args):
		self._syncImmediateValue()
		self.displayCommitted.emit(args)

	def _rawUiValue(self):
		"""return raw result from ui, without any processing
		so for a lineEdit, just text()"""
		raise NotImplementedError(self)

	def _setRawUiValue(self, value):
		"""set the final value of the ui to the exact value passed in,
		this runs after any pretty formatting"""
		raise NotImplementedError(self)

	def rxValue(self)->rx:
		return self._value
	def value(self)->T.Any:
		"""return widget's value as python object"""
		return self._value.rx.value

	def valueProxy(self) -> WpDexProxy:
		#return self._proxy
		return EVAL(self._proxy)

	def dex(self) -> WpDex:
		return EVAL(self._dex)

	def setValue(self, value:(WpDex, WpDexProxy, WX, T.Any)):
		"""
		TODO: should distinguish between setting the literal value pointed at,
			and setting/supplanting the current WpDex / proxy / WX references
		set value on widget - can be called internally and externally
		if passed a reactive element, set up all children and local
		attributes

		control flow goes :

		setvalue
			tryCommitValue
				commitValue
					rx.value set
						_syncUiFromValue()
							blockSignals()
								self.value()
									->processValueForUi()
										->setRawUiValue()
					syncImmediateValue
					valueCommitted qt signal
						-> connected to parent widgets, if any

		onDisplayCommitted
			rawResultFromUi()
				-> processResultFromUi()
					-> tryCommitValue()

		i guess when you put it like that it seems a bit complicated
		"""
		a = 1
		#log("set value", value, type(value), self)

		# the wrapped functions don't count as instances of rx

		if not (isinstance(value, (WpDex, WpDexProxy, WX, rx))
		        or react.getRx(value)) : # simple set value op
			try:
				check = (value == self.value())
				if check:
					return
			except: # if error raised, assume new value is not equal
				pass
			self._tryCommitValue(value)
		if isinstance(value, WpDex):
			self._dex = value
			self._proxy = value.getValueProxy()
			self._value = self.dex().ref()
			self._tryCommitValue(value.obj)
		elif isinstance(value, WpDexProxy):
			self._dex = value.dex()
			self._proxy = value
			self._tryCommitValue(value)
		elif isinstance(value, (WX, rx)) or react.getRx(value): # directly from a reference
			self._dex = lambda : value.RESOLVE(dex=True)
			self._proxy = lambda : value.RESOLVE(proxy=True)
			self._value = value
			# since it's an rx component, directly supplant the reactive value reference
			# self._value = value
			self._value.rx.watch(self._syncUiFromValue, onlychanged=False)
			log("before commit wx", value, EVAL(value))
			self._tryCommitValue(EVAL(value))


	def buildChildWidgets(self):
		raise NotImplementedError(self)


	def _onChildAtomicValueChanged(self,
	                               key:WpDex.pathT,
	                               value:T.Any,
	                               ):
		"""
		no fancy logic right now - if one of this widget's children change,
		rebuild everything
		check if new value needs new widget type -
		if so, remove widget at key and generate a new one

		cyclic link with _setAtomicChildWidget above, but I think it's ok

		TODO: resolve the case of modifying dict keys -
			it happens because we trigger write() with both the direct
			proxy reference, and this function.
			Once everything else holds together, disable all the
			try-excepts here and track it dowb

		"""
		#log("on child atomic widget changed", key, value, self)
		try: # if equality has been implemented for values, compare
			if value == self.dex().access(self.dex(), key, values=True):
				return
		except TypeError: #otherwise just write all the time to be safe
			pass
		except KeyError:
			pass
		try:
			self.dex().access(self.dex(), key, values=False).write(value)
		except (KeyError, Pathable.PathKeyError):
			pass # beyond caring

		self.buildChildWidgets()
		return

		#assert key in self._childAtomics, "no "

		oldDex = self._childAtomics[key].dex()
		newDexType = WpDex.adaptorForType(type(value))

		# if the old dex still fits the new type, nothing needs to be done
		if isinstance(oldDex, newDexType):
			return

		# make a new dex widget - pass in a reference to the branch of this dex
		# at the given key
		newChildWidget = self._makeNewChildWidget(
			key,
			self.dex().ref(*key),
			newDexType)
		self._setChildAtomicWidget(key, newChildWidget)



		"""
		if new value type not compatible:
		new widget = adaptor for value
		self.seyChildAtomic(newWidget)
		"""
	def _makeNewChildWidget(self,
	                        key,
	                        value,
	                        newDexType):
		newWidgetType: type[AtomicWidgetOld] = AtomicWidgetOld.adaptorForType(newDexType)
		return newWidgetType(value=value, parent=self)


	def rxImmediateValue(self)->rx:
		return self._immediateValue
	def immediateValue(self):
		return self._immediateValue.rx.value

	def _processResultFromUi(self, rawResult):
		"""override to do any conversions from raw ui representation"""
		return rawResult

	def _processValueForUi(self, rawValue):
		"""any conversions from a raw value to a ui representation"""
		return rawValue

	def _onDisplayEdited(self, *args, **kwargs):
		"""connect to base widget signal to update live -
		an error thrown here shows up as a warning,
		maybe change text colours"""
		#self._immediateValue.rx.value = self._processResultFromUi(self._rawUiValue())
		if self.warnLive:
			self._checkPossibleValue(self._immediateValue.rx.value
				)

	def _onDisplayCommitted(self, *args, **kwargs):
		"""connect to signal when user is happy, pressed enter etc"""
		self._tryCommitValue(
			self._processResultFromUi(self._rawUiValue())
		)
		self.clearFocus()

	def _checkPossibleValue(self, value):
		try:
			Condition.checkConditions(value, self.conditions)
		except Exception as e:
			log(f"Warning from possible value {value}:")
			traceback.print_exception(e)
			return

	def _tryCommitValue(self, value):
		"""TODO:
			calling _commitValue() directly in this prevents us from
			using super() in inheritor classes before extending the checks -
			find a better way to split the validation step up"""
		try:
			Condition.checkConditions(value, self.conditions)
		except Exception as e:
			# restore ui from last accepted value
			self._syncUiFromValue()
			log(f"ERROR setting value {value}")
			traceback.print_exc()
			return
		self._commitValue(value)


	def _syncUiFromValue(self, *args, **kwargs):
		"""update the ui to show whatever the current value is
		TODO: see if we actually need to block signals here"""
		#log("syncUiFromValue", self)

		# block signals around ui update
		self.blockSignals(True)
		toSet = self._processValueForUi(self.value())
		#log("_sync", toSet, type(toSet))
		self._setRawUiValue(self._processValueForUi(self.value()))
		self.blockSignals(False)

	def _commitValue(self, value):
		"""run finally after any checks, update held value and trigger signals"""
		# if widget is expressly enabled, we catch the error from setting value
		# on a non-root rx
		canBeSet = react.canBeSet(self._value)

		#log("root", self._value._root is self._value, self._value._compute_root() is self._value)
		#log("can be set", canBeSet, self)
		if canBeSet:
			value = EVAL(value)
			#log("setting", value)
			self._value.rx.value = value # rx fires ui sync function
		#log("after value set", self.value())
		self._syncImmediateValue()
		# self.syncLayout()
		#log("has valueCommitted", self, hasattr(self, "valueCommitted"))
		if hasattr(self, "valueCommitted"):
			self.valueCommitted.emit(value)


class AtomicWidgetOld(
	Adaptor,
	AtomicUiInterface

                   ):
	"""
	specialising reactive interface for full widgets and QObjects
	"""
	# atomic widgets registered as adaptors against dex types
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes : tuple[type[WpDex]] = ()

	Condition = Condition


	"""connect up signals and filter events in real widgets as
	needed to fire these - once edited and committed signals fire,
	logic should be uniform"""
	### COPY PASTE this block of signals to atomic classes that use it,
	### inheritance is getting too tangled here
	# widget display changed live
	displayEdited = QtCore.Signal(object)

	# widget display committed, enter pressed, etc
	# MAY NOT yet be legal, fires to start validation process
	displayCommitted = QtCore.Signal(object)

	# committed - FINAL value, no delta
	valueCommitted = QtCore.Signal(object) # redeclare this with proper signature

	valueType = object

	#TODO: context menu, action to pprint current value of dex

	def __init__(self, value:valueType=None,
	             conditions:T.Sequence[Condition]=(),
	             warnLive=False,
	             commitLive=False,
	             enableInteractionOnLocked=False
	             ):
		"""
		if commitLive, will attempt to commit as ui changes, still ignoring
		any that fail the check -
		use this for fancy sliders, etc
		"""
		AtomicUiInterface.__init__(self, value=value, conditions=conditions,
		                           warnLive=warnLive, commitLive=commitLive,
		                           enableInteractionOnLocked=enableInteractionOnLocked)


		self.displayEdited.connect(self._onDisplayEdited)
		self.displayCommitted.connect(self._onDisplayCommitted)

		self._value.rx.watch(self._syncUiFromValue, onlychanged=False)

		"""check that the given value is a root, and can be set -
		if not, disable the widget, since otherwise RX has a fit
		"""
		if not react.canBeSet(self._value):
			if not enableInteractionOnLocked:
				log(f"widget {self} passed value not at root, \n disabling for display only")
				self.setEnabled(False)

		self.setAutoFillBackground(True)

	def __post_init__(self, *args, **kwargs):
		"""yes I know post-inits are terrifying in qt,
		for this one, just call it manually yourself
		from the __init__ of the final class,
		i'm not your mother
		"""
		#log("postInit", self, self.value())
		#log(self._processValueForUi(self.value()))
		self._syncUiFromValue()
		# try: self._syncUiFromValue()
		# except: pass
		self._syncImmediateValue()
		self.syncLayout()



	def focusOutEvent(self, event):
		"""ensure we don't get trailing half-finished input left
		in the widget if the user clicks off halfway through editing -
		return it to whatever we have as the value"""
		self._syncUiFromValue()
		super().focusOutEvent(event)

	def syncLayout(self, execute:bool=True):
		"""sync layout of the table"""
		#log("syncLayout", self)
		# for i in self._childAtomics.values():
		# 	i.syncLayout(execute=execute)
		self.update()
		self.updateGeometry()
		if isinstance(self, QtWidgets.QAbstractItemView):
			self.scheduleDelayedItemsLayout()
			if execute:
				self.executeDelayedItemsLayout()
		self.update()
		self.updateGeometry()
		if isinstance(self.parent(), AtomicWidgetOld):
			self.parent().syncLayout(execute)


	# def sizeHint(self):
	#

class AtomicStandardItemModel(
	#inheritance.MetaResolver,
	AtomicUiInterface,
	QtGui.QStandardItemModel,
	Adaptor,
	metaclass=inheritance.resolveInheritedMetaClass(
		AtomicUiInterface, QtGui.QStandardItemModel, Adaptor
	)
):
	"""link to qobjects when using standardItems
	seems we're fully reverting to the old way,
	chaining item->view widget->model->item for each
	dex layer
	no idea how this fits together with all the single-widget
	fields for strings and assets (which is already working) -
	MAYBE we shift all our custom drawing to fancy delegates? to
	add extra buttons? and we still use a separate exp widget
	for editing.
	seems EXTREMELY complicated :(

	it's the contiguous selection within the model that scares
	me, I don't know where you would start emulating it if you had multiple
	layers of separate widgets

	it turns out splitting things up this much leads to
	unnervingly clean code.
	I guess if the complexity isn't in the logic, it's in the
	object structure itself
	"""

	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = ()
	# widget display changed live
	displayEdited = QtCore.Signal(object)

	# widget display committed, enter pressed, etc
	# MAY NOT yet be legal, fires to start validation process
	displayCommitted = QtCore.Signal(object)

	# committed - FINAL value, no delta
	valueCommitted = QtCore.Signal(object)  # redeclare this with proper signature

	modelsChanged = QtCore.Signal(object)

	def __init__(self, value, parent=None):
		QtGui.QStandardItemModel.__init__(self, parent)
		AtomicUiInterface.__init__(
			self, value=value)

		self.dataChanged.connect(self._onDataChanged)

	def __post_init__(self, *args, **kwargs):
		"""immediateValue kept here on the absolute extreme
		chance that we ever implement immediate structure changing
		to visualise as you drag items around in model views"""
		self.build()
		#self._syncUiFromValue()
		self._syncImmediateValue()

	def data(self, index, role:QtCore.Qt.ItemDataRole=...):
		"""Why do they query the model for front-end things like
		the visual alignment of text in the widget?
		Why do they do this???"""
		if role == QtCore.Qt.TextAlignmentRole:
			return QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
		return super().data(index, role)

	def _onDataChanged(self, *args, **kwargs):
		"""
		- block signals before and after to stop infinites
		- clear out items and models
		- rebuild
		- signal view SOMEHOW to redraw widgets - either custom signal or something like
			rowsReset, rowsAboutToBeReset etc

		TODO: if we edit multiple items, only fire this once somehow
		"""

		self.build()
		#self.modelsChanged.emit(self)


	def pathItemMap(self)->dict[WpDex.pathT, AtomStandardItem]:
		pathMap = {}
		for item in libmodel.iterAllItems(model=self):
			#item = self.itemFromIndex(index)
			if not isinstance(item, AtomStandardItem):
				continue
			pathMap[tuple(item.dex().relativePath(self.dex()  ))] = item
		return pathMap

	def childModels(self)->list[AtomicStandardItemModel]:
		return [i for i in self.children() if isinstance(i, AtomicStandardItemModel)]

	def pathModelMap(self)->dict[WpDex.pathT, AtomicStandardItemModel]:
		return {tuple(i.dex().relativePath(self.dex())) : i for i in self.childModels()}

	def build(self):
		self.blockSignals(True)
		self._buildItems()
		self._buildChildModels()
		self.blockSignals(False)
		self.modelsChanged.emit(self)
	def _buildItems(self):
		""" OVERRIDE
		create standardItems for each branch of
		this atom's dex
		for each one, check if it's a container - if so, make
		its own child view widget

		models built out in this function - VIEW has to INDEPENDENTLY
		construct and match up child view widgets for
		each entry that needs them?
		that might actually be the least complicated
		"""
		self.clear()
		for k, dex in self.dex().branchMap().items():
			itemType = AtomStandardItem.adaptorForObject(dex)
			item = itemType(value=dex)
			self.appendRow([item])


	def _buildChildModels(self):
		for i in self.children():
			if isinstance(i, AtomicStandardItemModel):
				i.deleteLater()
				i.setParent(None)
		for item in libmodel.iterAllItems(model=self):
			if not isinstance(item, AtomStandardItem):
				continue
			modelType = AtomicStandardItemModel.adaptorForObject(item.dex())
			if modelType: # add a child model and build it (maybe build should be done in init)
				newModel = modelType(value=item.dex(),
				                     parent=self)

	def _modelIndexForKey(self, key:WpDex.pathT)->QtCore.QModelIndex:
		key = WpDex.toPath(key)
		return self.model().index(int(key[0]), 1)

class AtomStyledItemDelegate(QtWidgets.QStyledItemDelegate,
                             #AtomicUiInterface,
                             Adaptor):
	"""base delegate only used to control which widget is
	displayed for editing -
	override for custom painting
	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (WpDex,)

	def paint(self, painter, option, index):
		"""sketch on how to specialise painting even if we have
		to use abstract type for delegate in general
		"""
		item = index.model().itemFromIndex(index)
		if not isinstance(item, AtomStandardItem):
			return QtWidgets.QStyledItemDelegate.paint(
				self, painter, option, index)
		delegateType = AtomStyledItemDelegate.adaptorForObject(item.dex())
		# avoid infinite loops - just paint normally
		if delegateType == AtomStyledItemDelegate:
			return QtWidgets.QStyledItemDelegate.paint(
				self, painter, option, index)
		return delegateType().paint(
			painter, option, index)

	def createEditor(self,
	                 parent:QtWidgets.QTreeView,
	                 option:QtWidgets.QStyleOptionViewItem,
	                 index:QtCore.QModelIndex):
		from wpdex.ui.atomic.expatom import ExpWidget
		log("delegate createEditor")

		item : AtomStandardItem = index.model().itemFromIndex(index)
		# until there's a case where we need multiple items per dex?
		assert isinstance(item, AtomStandardItem)
		assert not item.dex().branches, f"Item {item} is a container, can't directly edit it"
		return ExpWidget(value=item.dex(), # exp widget on the dex should be enough
		                 parent=parent)

	def trailWidgetsForItem(self,
	                        parent: QtWidgets.QTreeView,
	                        option: QtWidgets.QStyleOptionViewItem,
	                        index: QtCore.QModelIndex)->list[QtWidgets.QWidget]:
		"""sketch for how we could define other small widgets after this item;
		could do exactly the same thing for leading widgets.
		This would be used for file system navigation, for example - maybe
		version display for assets, etc
		"""

class AtomStandardItem(
AtomicUiInterface,
QtGui.QStandardItem,
Adaptor,
metaclass=inheritance.resolveInheritedMetaClass(
	AtomicUiInterface, QtGui.QStandardItem, Adaptor
                       )
    ):
	"""when you think you've fought through every circle of hell
	you just get another circle

	couldn't work out a good way of multi-selecting entries
	in a view when each one had its own widget, and also had
	more difficulties with tree views.

	so we try this way again - represent each leaf value by a
	special standard item that runs the normal atomic validation
	and links to the reactive references of its target value

	and maybe this can be made general enough that it supercedes all the
	other widget stuff I've done

	I swear I just want to make films

	I'd rather go fully one direction or another, but we can't even do that;
	seems like QTreeView doesn't properly display child item hierarchies from
	anything other than column 0, so we can't only use souped-up standardItems -
	we still need the separate widgets for separate containers.

	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (WpDex, )
	def __init__(self, value):
		QtGui.QStandardItem.__init__(self)
		AtomicUiInterface.__init__(
			self, value=value)
	def __post_init__(self, *args, **kwargs):
		self._syncUiFromValue()
		self._syncImmediateValue()

	def _syncUiFromValue(self, *args, **kwargs):
		self.setText(toStr(self.value()))

class AtomicView(QtWidgets.QTreeView,
                 Adaptor):
	"""customise view to control cursor motion,
	but logic of building shouldn't need to change.

	Views and models don't naturally include each other in
	parent chains - models directly parent models,
	and views directly parent their child widgets.
	Really hope this doesn't cause problems.

	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (WpDex, )

	if T.TYPE_CHECKING:
		def model(self)->AtomicStandardItemModel: ...
	def __init__(self, value:WpDex, parent=None,
	             model:AtomicStandardItemModel=None):
		QtWidgets.QTreeView.__init__(self, parent)
		# child widgets of container may not be direct children in Qt
		self._childAtomics : dict[WpDex.pathT, AtomicWindow] = WeakValueDictionary()
		if not model:
			modelType = AtomicStandardItemModel.adaptorForType(value)
			assert modelType, f"No atomic model type found for {value}, {type(value)}"
			model = modelType(value=value, parent=self)

		self.setItemDelegate(
			AtomStyledItemDelegate(parent=self))

		self.setModel(model)
		self.setAutoFillBackground(False)
		self.buildChildWidgets()

		self.model().modelsChanged.connect(self._onModelsChanged)

		# appearance and layout
		self.header().setDefaultSectionSize(2)
		#self.header().setMinimumSectionSize(-1) # sets to font metrics, still buffer around it
		self.header().setMinimumSectionSize(15)
		self.header().setSectionResizeMode(
			self.header().ResizeToContents
		)
		#self.setColumnWidth(0, 2)
		self.setIndentation(12)

		self.setAlternatingRowColors(True)
		self.setSizeAdjustPolicy(
			QtWidgets.QAbstractItemView.AdjustToContents)
		self.setHeaderHidden(True)

		self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
		self.setHorizontalScrollMode(self.ScrollMode.ScrollPerPixel)
		self.setContentsMargins(0, 0, 0, 0)
		self.setViewportMargins(0, 0, 0, 0)

		self.setUniformRowHeights(False)

	def _onModelsChanged(self, *args, **kwargs):
		"""easiest solution here is to just rip out all
		the models under item, rebuild them
		and rebuild child widgets afterwards"""
		self.buildChildWidgets()

	def dex(self):
		return self.model().dex()

	def buildChildWidgets(self):

		for i in self.children():
			if isinstance(i, (AtomicView, AtomicWindow)):
				i.close()
				i.deleteLater()
				#i.setParent(None)


		# set up index widgets on container dex items
		pathItemMap = self.model().pathItemMap()
		pathModelMap = self.model().pathModelMap()
		#log("path item map", pathItemMap)
		#log("pathModelMap", pathModelMap)
		for path, model in pathModelMap.items():
			item = pathItemMap[path]
			widget = AtomicWindow.adaptorForObject(item.dex())(
				value=item.dex(), parent=self,
				model=model)
			self.setIndexWidget(item.index(), widget)
			self._childAtomics[path] = widget

		self.setItemDelegate(  # use single type, manage dispatching from inside it
			AtomStyledItemDelegate(parent=self))

		for item in libmodel.iterAllItems(model=self.model()):
			self.setExpanded(item.index(), True)

	def syncLayout(self):
		log("sync layou")
		self.updateGeometries()
		self.scheduleDelayedItemsLayout()
		self.executeDelayedItemsLayout()
		self.updateGeometries()

class ViewExpandButton(QtWidgets.QPushButton):
	"""button to show type of container when open,
	and overview of contained types when closed"""
	expanded = QtCore.Signal(bool)
	def __init__(self, openText="[", dex:WpDex=None, parent=None):
		self._isOpen = True
		self._openText = openText
		self._dex = dex
		super().__init__(openText, parent=parent)

		m = 0
		self.setContentsMargins(m, m, m, m)
		self.setFixedSize(13, 20,  )
		self.setStyleSheet("padding: 1px 1px 2px 2px; text-align: left")

		self.clicked.connect(lambda : self.setExpanded(
			state=(not self.isExpanded()), emit=True))

	def getClosedText(self):
		return self._dex.getTypeSummary()

	def setExpanded(self, state=True, emit=False):
		log("setExpanded", state, emit)
		if state:
			self.setText(self._openText)
			self.setFixedSize(13, 20, )
		else:
			self.setText(self.getClosedText())
			self.setMaximumWidth(100)
		self._isOpen = state
		if emit:
			self.expanded.emit(state)
	def isExpanded(self):
		return self._isOpen

class AtomicWindow(ContextMenuProvider,
                   #QtWidgets.QWidget,
                   QtWidgets.QFrame,
                   Adaptor,

                   ):
	"""it wasn't complex enough
	overall holder widget to contain separate widgets alongside each view -
	specifically the expand/contract button,

	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (WpDex, )

	def dex(self)->WpDex:
		return self.view.dex()

	def atomicViewParent(self)->AtomicView:
		if self.parent() is None: return None
		return self._origParent
	def atomicMainParent(self)->AtomicWindow:
		if self.parent() is None: return None
		return self._origParent.parent()


	def __init__(self, value:WpDex, parent=None,
	             model:AtomicStandardItemModel=None):
		#QtWidgets.QWidget.__init__(self, parent)
		QtWidgets.QFrame.__init__(self, parent)
		self._origParent = parent # VERY BAD but necessary to survive through setIndexWidget()
		self.setLayout(QtWidgets.QHBoxLayout(self))
		value = getWpDex(value)
		self.view = AtomicView.adaptorForObject(value)(
			value=value, parent=self, model=model)

		self.expandBtn = ViewExpandButton(
			openText=self.dex().bookendChars()[0],
			dex=self.dex(),
			parent=self
		)
		self.expandBtn.expanded.connect(self._onExpandBtnClicked)
		self.layout().addWidget(self.expandBtn)
		self.layout().addWidget(self.view)

		self.setContentsMargins(0, 0, 0, 0)
		self.layout().setContentsMargins(0, 0, 0, 0)
		#self.layout().setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
		#self.layout().setAlignment(self.expandBtn, QtCore.Qt.AlignTop)
		self.layout().setAlignment(self.expandBtn, QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
		self.setAutoFillBackground(True)

		ContextMenuProvider.__init__(self, value=value, parent=parent)

	def _isExpanded(self)->bool:
		return self.view.isVisible()

	def _onExpandBtnClicked(self):
		"""consider replacing view with raw Exp widget showing
		string representation of value, and ALLOWING EDITING?
		"""
		if self.expandBtn.isExpanded():
			self.view.show()
			#self.layout().addWidget(self.view)
		else:
			self.view.hide()
			#self.layout().removeWidget(self.view)
		if self.atomicViewParent():
			self.atomicViewParent().syncLayout()
			#self.parent().updateGeometry()


	def _getBaseContextTree(self, *args, **kwargs) ->Tree[str, callable]:
		tree = Tree("contextMenuTree")
		tree["display"] = lambda : pprint.pprint(serialise(self.dex().obj))
		return tree

	# def contextMenuEvent(self, event:QtGui.QContextMenuEvent):
	# 	menu = treemenu.buildMenuFromTree(self.menuTree)
	# 	self._menu = menu # setting menu's parent gave weird behaviour -
	# 	# this way we still hold a reference to it so it doesn't vanish
	# 	menu.move(event.globalPos())
	# 	menu.show()






