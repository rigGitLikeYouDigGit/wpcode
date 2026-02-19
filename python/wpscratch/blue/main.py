from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from __future__ import annotations

import json
import os
from pathlib import Path
import types, typing as T
from Qt import QtCore, QtWidgets, QtGui
from ..asset import Asset
from ..plan import Plan
from .plan_widget import PlanWidget
from .asset_selector import AssetSelector
from .task_widget import TaskWidget

"""
main window for Blue rigging system

TODO: need to stop key presses and events leaking through to main maya window 
"""
def get_show_from_path(cwdPath):
    p = Path(cwdPath)
    print("path", p, "parts", p.parts)
    parts = p.parts
    if parts[0] == "/":
        parts = parts[1:]
    if (len(parts) < 2):
        return None
    if parts[0] != "jobs":
        return None

    return parts[1]



class BlueWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.asset_w = AssetSelector(self)
        self.plan_w = PlanWidget(self)
        self.task_w = TaskWidget(self)

        self.save_btn = QtWidgets.QPushButton("Save", self)
        self.load_btn = QtWidgets.QPushButton("Load", self)
        self.new_btn = QtWidgets.QPushButton("New", self)

        # might change up ways of interacting here - would we ever want to reset and not build to either guides or rig?
        self.reset_btn = QtWidgets.QPushButton("Reset", self)
        self.guide_btn = QtWidgets.QPushButton("Build Guides", self)
        self.rig_btn = QtWidgets.QPushButton("Build Rig", self)


        self.setWindowTitle("BLUE")
        self.makeLayout()
        self.makeConnections()

        # retrieve previous settings to focus last opened asset
        self.settings = QtCore.QSettings("Untold", "RigPlanner")
        print("get last asset")
        found = self.settings.value("asset", None)
        print("found", found)

        self.focus_last_asset()
        self.setStyleSheet("background:rgba(30,30,70, 100);")

    def asset(self)->Asset:
        return self.asset_w.get_asset()

    def plan(self)->Plan:
        return self.plan_w.plan

    def makeLayout(self):
        vl = QtWidgets.QVBoxLayout(self)
        vl.addWidget(self.asset_w)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.new_btn)
        hl.addWidget(self.load_btn)
        hl.addWidget(self.save_btn)
        vl.addLayout(hl)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.reset_btn)
        hl.addWidget(self.guide_btn)
        hl.addWidget(self.rig_btn)
        vl.addLayout(hl)

        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.plan_w)
        hl.addWidget(self.task_w)
        vl.addLayout(hl)


        self.setLayout(vl)

    def makeConnections(self):
        self.plan_w.task_selected.connect(
            self._on_tasks_selected
        )
        self.asset_w.assetChanged.connect(self._on_asset_changed)

        self.load_btn.pressed.connect(self._on_load_pressed)
        self.save_btn.pressed.connect(self._on_save_pressed)
        self.reset_btn.pressed.connect(self._on_reset_pressed)
        self.guide_btn.pressed.connect(self._on_guides_pressed)
        self.rig_btn.pressed.connect(self._on_rig_pressed)

    def _on_tasks_selected(self, data:dict):
        if data["tasks"]:
            self.task_w.setTask(data["tasks"][0])

    def _on_asset_changed(self, tokens:list):
        print("on_asset_changed", tokens)
        if self.asset_w.get_asset().exists():
            self.save_btn.setEnabled(True)
            self.plan_w.plan.set_asset(self.asset_w.get_asset())
        else:
            self.save_btn.setEnabled(False)
        if self.asset_w.get_rig_path() and self.asset_w.get_rig_path().exists():
            self.load_btn.setEnabled(True)
        else:
            self.load_btn.setEnabled(False)
        self.settings.setValue("show/lastShow", tokens[0])
        self.settings.setValue("show/{}/lastAsset".format(tokens[0]), tokens)


    def _on_save_pressed(self, *args, **kwargs):
        """no fancy save plan/save scene/save data push/pull martial arts katas yet,
        just save current state of plan to target file
        we also have no version control with this, so exercise discipline
        """
        data_to_save = self.plan_w.plan.to_dict()
        self.asset_w.get_rig_path().write_text(
            json.dumps(data_to_save, indent=4)) # do we want json

    def _on_load_pressed(self,  *args, **kwargs):
        assert self.asset_w.get_rig_path().exists()
        data_to_load = json.loads(self.asset_w.get_rig_path().read_text())
        asset = self.asset()
        newPlan = Plan.from_dict(data_to_load, asset)
        self.plan_w.set_plan(newPlan)

    def _on_reset_pressed(self, *args, **kwargs):
        self.plan().reset()

    def _on_guides_pressed(self, *args, **kwargs):
        self.plan().reset()
        self.plan().exec(0)
    def _on_rig_pressed(self, *args, **kwargs):
        self.plan().exec(1)

    def focus_last_asset(self):

        lastShow = get_show_from_path(os.getcwd())
        print("currentShow", lastShow)
        if lastShow is None: # maya not started from within show dir, take previous asset
            lastShow = self.settings.value("show/lastShow")
            print("found lastShow", lastShow)
            if lastShow is None:
                print("could not find last show or asset, exiting")
                return
        lastAsset = self.settings.value("show/" + lastShow + "/lastAsset")
        print("found lastAsset", lastAsset)
        if lastAsset is None:
            self.asset_w.setTokens([lastShow])
            return
        self.asset_w.setTokens(lastAsset)
        return

    # def event(self, *args, **kwargs):
    #     super().event(*args, **kwargs)
    #     return True

