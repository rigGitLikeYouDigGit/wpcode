from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from __future__ import annotations
import typing as T, types
import importlib, traceback
from pathlib import Path


from ..lib.code_ref import CodeRef
from ..lib.catalogue import TypeCatalogue
from ..lib.signal import Signal
from ..asset import Asset

from maya import cmds
import maya.api.OpenMaya as om

if T.TYPE_CHECKING:
    from ..plan import Plan
"""holder for individual operation in maya -

can be subclassed alone, or used to wrap pre-existing rig_build modules
as we grow
"""

class Task:
    """
    base class for rigging operations
        """

    colour = (0.5, 0.5, 0.7)

    task_catalogue : TypeCatalogue = None

    ABSTRACT = False

    def __init__(self, name:str="", plan:Plan=None):
        self.name = name or self.__class__.__name__
        self.plan : Plan = plan # set when task is added to plan
        self.task_changed = Signal("task_changed")
        self.builtStage = -1
        self.runningStage = -1
        self.params = {}
        self.set_param_defaults(self.params)

    def set_name(self, name:str):
        self.name = name
        print("task set_name", name)
        self.task_changed.emit(self)

    def get_asset(self)->Asset:
        if self.plan is None: return None
        return self.plan.get_asset()

    def set_param_defaults(self, param_dict:dict):
        """modify and update input param dict with default params
        maybe we could later pass in previous params, when loaded from file-
        for now just use a fresh one
        """


    def guide(self):
        """ OVERRIDE THIS
        user-override method for building guides"""
        print("Base task guide :D")

    def run_guide(self):
        """ DO NOT OVERRIDE THIS
        wrapper method for running guide and cleaning up later
        """
        self.runningStage = 0
        self.guide()
        self.runningStage = -1
        self.builtStage = 0

    def build(self):
        """ OVERRIDE THIS
        user-override method for building main rig functionality"""
        print("Base task build :D")

    def run_build(self):
        """ DO NOT OVERRIDE THIS
        wrapper method for build and cleanup
        """
        self.runningStage = 1
        self.build()
        self.runningStage = -1
        self.builtStage = 1

    def stageFunctions(self)->dict[str, types.FunctionType]:
        """maybe a bit overengineered
        if your task needs extra steps, add them in here -
        should we add in some way to insert tasks in the midpoint,
        should we have a float-based system instead to weight them somehow,
        idk
        """
        return {"guide" : self.run_guide,
                "build" : self.run_build
                }

    def reset(self):
        """delete any nodes associated with this task
        """
        self.builtStage = -1
        self.runningStage = -1

    # maya stuff

    # maya stuff
    def rig_grp(self):
        if not cmds.objExists("rig_GRP"):
            cmds.createNode("transform", n="rig_GRP")
        return "rig_GRP"
    def model_grp(self):
        if not cmds.objExists("model_GRP"):
            grp = cmds.createNode("transform", n="model_GRP")
            cmds.parent(grp, self.rig_grp())
        return "model_GRP"



    ##### SERIALISATION
    def to_dict(self)->dict:
        """super simple for now -
        we can go VERY hard on serialisation in the future, but for now it's just a single layer for params
        """
        return {
            "@T" : CodeRef.get(self),
            "name" : self.name,
            "params" : self.params
        }

    @classmethod
    def from_dict(self, d:dict, plan:Plan)->Task:
        """ DO NOT OVERRIDE THIS
        given a task data dict of unknown type, deserialise it
        and return the rehydrated object.
        Again, we can go much further with automated reconstruction of
        complex hierarchies, and it also avoids the inelegant public/internal methods below
        """
        if not "@T" in d:
            raise RuntimeError("Task data dict has no type key @T, cannot retrieve task type")
        task_t : type[Task] = CodeRef.resolve(d["@T"])
        print("resolved task type:", task_t)
        return task_t._from_dict_internal(d, plan)
    @classmethod
    def _from_dict_internal(cls, d:dict, plan:Plan)->Task:
        """OVERRIDE THIS
        any task-specific logic to recover the task object"""
        task = cls(d["name"], plan=plan)
        task.params = d["params"]
        return task

# scan directories for task classes - easily extensible based on shows, configrules, etc
Task.task_catalogue = TypeCatalogue(
    Task, ["u_rig.tools.blue.task"],
    includeFn=lambda t : not getattr(t, "ABSTRACT", False)

)

