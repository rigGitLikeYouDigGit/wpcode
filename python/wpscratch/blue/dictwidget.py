from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from __future__ import annotations

import json
import typing as T, types
import importlib, traceback
from pathlib import Path

from Qt import QtCore, QtWidgets, QtGui

from ..lib.code_ref import CodeRef
from ..lib.catalogue import TypeCatalogue
from ..lib import exp
from ..asset import Asset

from ..plan import Plan
from ..task import Task


"""VERY VERY SIMPLE 
Qt dict editor for literal dict values only - 
we can and will do better here
"""

class DictWidget(QtWidgets.QTableView):

    dictChanged = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModel(QtGui.QStandardItemModel(self))

        self.model().itemChanged.connect(self._on_data_changed)

        #self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    if T.TYPE_CHECKING:
        def model(self)->QtGui.QStandardItemModel:...

    def _on_data_changed(self, *args, **kwargs):
        self.dictChanged.emit(self.get_dict())

    def set_dict(self, d:dict):
        self.model().clear()
        for k, v in d.items():
            self.model().appendRow(
                #[QtGui.QStandardItem(json.dumps(k)), QtGui.QStandardItem(json.dumps(v))]
                [QtGui.QStandardItem(exp.strings_to_names(str(k))),
                 QtGui.QStandardItem(exp.strings_to_names(str(v)))]
                #[QtGui.QStandardItem(str(k)), QtGui.QStandardItem(str(v))]
            )
    def get_dict(self)->dict:
        """TODO: better evaling, allow literal names, etc"""
        result = {}
        for i in range(self.model().rowCount()):
            # k = self.model().data(QtCore.QModelIndex(i, 0))
            # v = self.model().data(QtCore.QModelIndex(i, 1))
            k = self.model().data(self.model().index(i, 0))
            v = self.model().data(self.model().index(i, 1))
            # if(v[0] in "([{"):
            #     result[k] = eval(v)
            # else:
            #     result[k] = v
            #result[json.loads(k)] = json.loads(v or "")
            result[k] = eval(exp.names_to_strings(v))

        return result

    # maya integration - right click to copy selection
    def contextMenuEvent(self, arg__1:QtGui.QContextMenuEvent) -> None:
        index = self.indexAt(arg__1.pos())
        print("index at", index)
        if not index.isValid():
            return False
        if index.column() != 1:
            return False
        try:
            from maya import cmds
        except ImportError:
            print("not running in maya")
            return False
        menu = QtWidgets.QMenu()
        fromSceneAction = menu.addAction("paste scene nodes")

        fromSceneAction.triggered.connect(lambda *a, **kw : self.paste_scene_nodes(index))

        menu.exec_(arg__1.globalPos())

    def paste_scene_nodes(self, index:QtCore.QModelIndex):
        try:
            from maya import cmds
        except ImportError:
            print("not running in maya")
            return False
        base_data = eval(exp.names_to_strings(self.model().data(index)))
        #base_data = json.loads(self.model().data(index))
        #print("json loaded data", base_data)
        if not isinstance(base_data, list):
            base_data = [base_data]

        scene_nodes = cmds.ls(sl=1) or []
        for i in scene_nodes:
            base_data.append(i)
        self.model().setData(index, exp.strings_to_names(str(base_data)))


    def select_nodes_in_scene(self, index:QtCore.QModelIndex, event:QtGui.QContextMenuEvent):
        try:
            from maya import cmds
        except ImportError:
            print("not running in maya")
            return False
        base_data = eval(exp.names_to_strings(self.model().data(index)))

        if not isinstance(base_data, list):
            base_data = [base_data]

        scene_nodes = cmds.ls(base_data) or []

        #TODO: be a bit extra
        #   check if we need to toggle, add or remove from current selection
        # # if no modifiers
        # event.modifiers() & Qt.KeyboardModifier.ControlModifier
        # if event.modifiers()