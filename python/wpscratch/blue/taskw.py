from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from __future__ import annotations
import typing as T, types
import importlib, traceback
from pathlib import Path

from Qt import QtCore, QtWidgets, QtGui

from ..lib.code_ref import CodeRef
from ..lib.catalogue import TypeCatalogue
from ..asset import Asset

from ..plan import Plan
from ..task import Task

from .dict_widget import DictWidget


class TaskWidget(QtWidgets.QWidget):
	"""base class for UI representing a single task -
	allow specialising widgets where useful for
	specific tasks
	"""

	for_task = Task

	def __init__(self, parent=None, task: Task = None):
		super().__init__(parent)
		self.task: Task = None
		self.type_label = QtWidgets.QLabel("no task selected", self)  # TODO: task casting
		self.type_label.setEnabled(False)
		self.name_line = QtWidgets.QLineEdit(self)
		self.param_widget = DictWidget(parent=self)

		self.make_connections()
		self.make_layout()

		if task:
			self.setTask(task)

	def make_connections(self):
		self.name_line.returnPressed.connect(self._on_name_set)
		self.param_widget.dictChanged.connect(self._on_params_changed)
		pass

	def make_layout(self):
		vl = QtWidgets.QVBoxLayout(self)
		vl.addWidget(self.type_label)
		vl.addWidget(self.name_line)
		vl.addWidget(self.param_widget)
		self.setLayout(vl)

	def _on_name_set(self, *args, **kwargs):
		s = self.name_line.text()
		if not s:
			return
		self.task.set_name(s)

	def _on_params_changed(self, *args, **kwargs):
		self.task.params = self.param_widget.get_dict()

	def setTask(self, task: Task):
		"""re-sync whole widget - very basic for now
		"""
		self.task = task
		self.type_label.setText("<" + task.__class__.__name__ + ">")
		self.name_line.setText(task.name)
		self.param_widget.set_dict(task.params)
