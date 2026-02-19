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


class ModelTask(Task):
    """Import a versioned model into scene
        TODO: SUPER ROUGH, couldn't figure out how to get Stem api working with this -
            later maybe get stem ui into the task widget
    """

    def set_param_defaults(self, param_dict:dict):
        param_dict["tokens"] = self.get_asset().tokens()
        param_dict["version"] = -1 # to always pick up latest
        param_dict["direct_path"] = "" # if set, always overrides exact file

    def guide(self):

        self.filePath = Path(self.params["direct_path"])
        assert self.filePath.exists()

        with SceneDelta() as sd:
            cmds.file(str(self.filePath), i=1)
        for i in sd.assemblies:
            cmds.parent(i, self.model_grp())







