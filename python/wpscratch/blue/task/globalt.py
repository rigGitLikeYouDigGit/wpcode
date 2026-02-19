from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from __future__ import annotations
import importlib, traceback
from pathlib import Path
import typing as T, types
from u_rig.tools.rig_module.rig_module import RigModule

from u_rig.tools.rig_module.rig_hierarchy import RigHierarchy
from .base import Task


if T.TYPE_CHECKING:
    from ..plan import Plan


class GlobalTask(Task):
    """
   task to create global control
   TODO:register weights
    """

    def set_param_defaults(self, param_dict:dict):
        pass

    def guide(self):
        self.hierarchy = RigHierarchy()

        self.hierarchy.build_guides()

    def build(self):
        self.hierarchy.build_bones()
        self.hierarchy.build_rig()
        self.hierarchy.clean_up()

