from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from __future__ import annotations
import importlib, traceback
from pathlib import Path
from Qt import QtCore, QtWidgets, QtGui

from ..plan import Plan

from ..task import Task, RigModuleTask  # bit messy but worth it to call up rigmodules directly


class SearchableComboBox(QtWidgets.QComboBox):
	"""from stackOverflow"""

	def __init__(self, parent=None):
		super(SearchableComboBox, self).__init__(parent)

		self.setFocusPolicy(QtCore.Qt.ClickFocus)
		self.setEditable(True)

		# prevent insertions into combobox
		self.setInsertPolicy(QtWidgets.QComboBox.NoInsert)

		# filter model for matching items
		self.pFilterModel = QtCore.QSortFilterProxyModel(self)
		self.pFilterModel.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
		self.pFilterModel.setSourceModel(self.model())

		# completer that uses filter model
		self.completer = QtWidgets.QCompleter(self.pFilterModel, self)
		self.completer.setCompletionMode(QtWidgets.QCompleter.UnfilteredPopupCompletion)
		self.setCompleter(self.completer)

		# connect signals
		self.lineEdit().textEdited[str].connect(self.pFilterModel.setFilterFixedString)
		self.completer.activated.connect(self.on_completer_activated)

	def on_completer_activated(self, text):
		if text:
			index = self.findText(text)
			self.setCurrentIndex(index)
			self.activated[str].emit(self.itemText(index))

	def setModel(self, model):
		super(SearchableComboBox, self).setModel(model)
		self.pFilterModel.setSourceModel(model)
		self.completer.setModel(self.pFilterModel)

	def setModelColumn(self, column):
		self.completer.setCompletionColumn(column)
		self.pFilterModel.setFilterKeyColumn(column)
		super(SearchableComboBox, self).setModelColumn(column)


class TreeViewSelectionCursorInteractor(QtCore.QObject):
	"""slightly over-engineered way to clear view selection when
	you click on an empty region"""

	def eventFilter(self, watched: QtWidgets.QAbstractItemView, event):
		if not isinstance(event, QtGui.QMouseEvent):
			return super().eventFilter(watched, event)

		at = watched.indexAt(event.pos())
		print("at", at)
		if not at:
			watched.clearSelection()
		return super().eventFilter(watched, event)


class PlanWidget(QtWidgets.QTreeView):
	"""
	main widget to display blue scene and edit its structure
	go with simple list view and model for now, the simpler the better
	basic sync method, regenerate whole ui, the usual startpoint

	for now, check that we can map out different existing rig modules
	"""

	task_selected = QtCore.Signal(dict)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setModel(QtGui.QStandardItemModel(self))

		self.plan: Plan = None

		self.makeLayout()
		self.makeConnections()
		# self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
		self.viewport().installEventFilter(self)

		self.set_plan(Plan())

	# if I don't override event() like this, every mousePressEvent is transformed agent-smith-like
	# into a contextMenuEvent before it's even passed to this function
	def event(self, event: QtCore.QEvent) -> bool:
		# print("EVENT", event)
		return super().event(event)

	def eventFilter(self, object: QtCore.QObject, event: QtCore.QEvent) -> bool:
		# print("event filter", object, event)
		return super().eventFilter(object, event)

	def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
		# print("mouse press event", event, event.type() == QtGui.QMouseEvent.Type.MouseButtonPress)
		# event = QtGui.QMouseEvent.
		if isinstance(event, QtGui.QContextMenuEvent):
			# return super().mousePressEvent(event)
			return True
		# print("mouse press event", event)
		if event.button() != QtCore.Qt.LeftButton:
			return super().mousePressEvent(event)
		at = self.indexAt(event.pos())

		if not at.isValid():
			self.clearSelection()
		return super().mousePressEvent(event)

	def makeLayout(self):
		# vl = QtWidgets.QVBoxLayout(self)
		#
		# vl.addWidget(self.view)
		# self.setLayout(vl)
		pass

	def makeConnections(self):
		pass

	def taskForIndex(self, index: QtCore.QModelIndex):
		"""TODO: more robust"""
		return self.plan.model[index.row()]

	def indexForTask(self, task: Task):
		if not task in self.plan.model:
			return -1
		return self.plan.model.index(task)

	def selectionChanged(self, selected: QtCore.QItemSelection, deselected: QtCore.QItemSelection) -> None:
		super().selectionChanged(selected, deselected)
		self.task_selected.emit({"tasks": [self.taskForIndex(i) for i in self.selectedIndexes()]})

	def set_plan(self, plan: Plan):
		self.plan = plan
		self.model().clear()
		for i in self.plan.model:
			self.model().appendRow(QtGui.QStandardItem(i.name))
		self.plan.plan_changed.connect(self.on_plan_changed)

	def on_plan_changed(self, change: dict):
		"""connect to plan object signal, update UI whenever structure changed
		"""
		self.set_plan(self.plan)

	def get_context_menu(self, arg__1: QtGui.QContextMenuEvent) -> QtWidgets.QMenu:
		menu = QtWidgets.QMenu()
		menu.addAction("hello")
		return menu

	def get_task_menu(self, arg__1: QtGui.QContextMenuEvent) -> QtWidgets.QMenu:
		menu = QtWidgets.QMenu()

		box = SearchableComboBox(menu)
		model = QtCore.QStringListModel(box)
		# print("get task types:")
		model.setStringList(
			# list(Task.task_catalogue.pathTypeMap.keys())
			list(Plan.get_available_tasks())
		)
		# for k, v in Task.task_catalogue.pathTypeMap.items():
		#     print(k, v)
		#     model.insertRow(k)
		box.setModel(model)
		box.setPlaceholderText("task type...")

		vl = QtWidgets.QVBoxLayout()
		vl.setContentsMargins(0, 0, 0, 0)
		menu.setLayout(vl)
		menu.layout().addWidget(box)
		menu.setFixedWidth(400)

		box.setFocus()
		box.lineEdit().selectAll()

		# box.returnPressed.connect(lambda *args, **kwargs : self._on_task_submitted(menu, box.currentText()))
		# box.textActivated.connect(lambda *args, **kwargs : self._on_task_submitted(menu, box.currentText()))
		box.lineEdit().returnPressed.connect(lambda *args, **kwargs: self._on_task_submitted(menu, box.currentText()))
		return menu

	def _on_task_submitted(self, menu: QtWidgets.QMenu, text: str):
		result = self.plan.new_task_requested(text)
		if result is None:
			print("unknown task requested: " + text + " , skipping")
		menu.close()

	def contextMenuEvent(self, arg__1: QtGui.QContextMenuEvent) -> None:
		# print("context menu event")
		if (self.selectedIndexes()):
			w = self.get_context_menu(arg__1)
			w.exec_(arg__1.globalPos())
			return True
		# if nothing selected, allow creating new task
		line = self.get_task_menu(arg__1)
		line.exec_(arg__1.globalPos())
		return True
	# return super().contextMenuEvent()

