from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from __future__ import annotations
import importlib, traceback
from pathlib import Path
import typing as T, types
from u_rig.tools.rig_module.rig_module import RigModule
from .base import Task
from ..lib.catalogue import TypeCatalogue
from ..lib.code_ref import CodeRef

if T.TYPE_CHECKING:
    from ..plan import Plan

"""
holder for previous rigModule system's objects
"""


class RigModuleTask(Task):
    """wrapper task for rigModule objects-
    provide wrappers for setting defaults

    PARAMS for which attributes/weights to register

    unsure if we should bind live RigModule objects like this or
    only addresses, and regenerate new objects on fly during
    build -
    leaning toward the latter
    """


    rig_module_catalogue : TypeCatalogue = None

    ABSTRACT = True # don't show up directly in searches

    def __init__(
            self, rigModule:RigModule, name:str="", plan:Plan=None
                 ):
        self.rig_module = rigModule
        # supercede name to keep in sync
        name = name or rigModule.name
        rigModule.name = name
        super().__init__(name, plan=plan)

    def set_param_defaults(self, param_dict:dict):

        self.rig_module.set_defaults()
        param_dict.update(self.rig_module.defaults)

    def guide(self):
        self.rig_module.defaults = self.params
        self.rig_module.build_guides()

    def build(self):

        self.rig_module.build_bones()
        self.rig_module.build_rig()
        """TODO: intervene here to check any deformers
                created by deform, and add them to register
                """
        self.rig_module.deform()

    def to_dict(self)->dict:
        d = super().to_dict()
        d["rigModule"] = CodeRef.get(self.rig_module)
        return d

    @classmethod
    def _from_dict_internal(cls, d:dict, plan:Plan) ->Task:
        rm = CodeRef.resolve(d["rigModule"])(d["name"])
        task = cls(plan=plan,name=d["name"],  rigModule=rm)
        task.params = d["params"]
        return task

RigModuleTask.rig_module_catalogue = TypeCatalogue(
    RigModule,
    ["u_rig.tools.rig_module"]
)
