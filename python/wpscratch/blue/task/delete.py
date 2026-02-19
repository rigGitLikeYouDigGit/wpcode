from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


from __future__ import annotations
import typing as T, types
import importlib, traceback
from pathlib import Path


from ..asset import Asset
from ..task.base import Task

from ..maya.lib.scene import SceneDelta

from maya import cmds
import maya.api.OpenMaya as om

if T.TYPE_CHECKING:
    from tools.blue.plan import Plan

def make_list(v):
    return v if isinstance(v, (list, tuple)) else [v]
class DeleteTask(Task):
    """
    Delete wildcarded selections of nodes during either guide or rig stage
    """

    def set_param_defaults(self, param_dict:dict):
        param_dict["guide"] = []
        param_dict["rig"] = []

    def deleteExpValues(self, expRaw):
        exps = make_list(expRaw)
        for i in exps:
            nodes = cmds.ls(i) or []
            for n in nodes:
                if cmds.objExists(n):
                    try:
                        cmds.delete(n)
                    except Exception as e:
                        print("could not delete node:" + n)
                        continue

    def guide(self):
        self.deleteExpValues(self.params["guide"])
    def build(self):
        self.deleteExpValues(self.params["rig"])




