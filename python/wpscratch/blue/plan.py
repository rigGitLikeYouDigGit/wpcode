from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
from __future__ import annotations
import importlib, traceback
from pathlib import Path
from .task import Task, RigModuleTask
from u_rig.tools.rig_module.rig_module import RigModule

from .lib.signal import Signal
from .asset import Asset

from maya import cmds
from maya.api import OpenMaya as om

""" 
overall container for all tasks/modules in a rig, and their sequencing
for now, run all tasks sequentially - guides, then build

"""

"""

"""

class Plan:
    """TODO: convert model to tree,
        single level dict is fine for now

    Holds the "skeleton" of a rig build - task objects and execution order
    This object also acts as the global environment
    of the rig - it's not a singleton itself, in case we ever want to deal with multiple alongside

    TODO: maybe do proper delta types to describe changes in scene, but for now
        it's enough to reread the whole thing on any change

        """


    def __init__(self, name="rig"): # use names to differentiate different components at asset-level
        self.name = name
        self.model : list[Task] = []

        self._asset : Asset = None # integration with stem / asset system

        self.plan_changed = Signal("planChanged")

    def get_asset(self)->Asset:
        return self._asset

    def set_asset(self, asset:Asset):
        self._asset = asset

    def get_data_dir(self)->Path:
        asset = self.get_asset()
        if asset is None: return None
        return asset.scriptDir() / (self.name + "_data")

    def get_plan_path(self)->Path:
        asset = self.get_asset()
        if asset is None: return None
        return asset.scriptDir() / (self.name + ".plan")

    def reset(self):
        for k, task in (enumerate(reversed(self.model))):
            task.reset()
        cmds.file(new=1, f=1)

    def add_task(self, task:Task):
        self.model.append(task)
        task.plan = self
        task.task_changed.connect(lambda *args, **kwargs : self.plan_changed.emit({"added" : task}))
        self.plan_changed.emit({"added" : task})

    def remove_task(self, task:Task):
        if not task in self.model:
            return
        self.model.remove(task)
        self.plan_changed.emit({})

    def clear(self):
        """remove all tasks from model"""
        self.model.clear()
        self.plan_changed.emit({})


    def exec(self, toStage=-1):
        """Run entire rig build from scratch -
        FOR NOW reset scene, later see if we can selectively only delete nodes
        affected by build process
        """
        if(toStage < 0):
            self.reset()

        """TODO: add possibility for extra build steps, in exceptional tasks - no big hassle to do here"""
        # get max stage index
        maxStageIndex = 1
        for task in self.model:
            maxStageIndex = max(maxStageIndex, len(task.stageFunctions()))

        for i in range(0, maxStageIndex ):
            self.stageFns()[i]()
            for task in self.model:
                if(task.builtStage < i):
                    taskStageFn = tuple(task.stageFunctions().values())[i]
                    taskStageFn()

    def preGuide(self):
        """run function before building to guides"""
        cmds.file(new=1, f=1) # clear maya scene

    def preRig(self):
        pass

    def stageFns(self):
        return [self.preGuide, self.preRig]


    @staticmethod
    def get_available_tasks()->dict[str, type[Task]]:
        task_map = Task.task_catalogue.pathTypeMap
        task_map["----"] = Task
        task_map.update(RigModuleTask.rig_module_catalogue.pathTypeMap)
        return task_map

    def new_task_requested(self, request:str)->Task:
        task_type = self.get_available_tasks().get(request)
        if task_type is None:
            return None
        if issubclass(task_type, RigModule ):
            new_task = RigModuleTask(task_type(task_type.__name__), "")
            self.add_task(new_task)
            return new_task
        new_task = task_type(plan=self)
        self.add_task(new_task)
        return new_task

    #### SERIALISATION
    def to_dict(self)->dict:
        """return serialised plan to save to file"""
        data = {"name" : self.name,

                "model" : []} # unsure if we should pack rig names into plan files
        for v in self.model:
            data["model"].append( v.to_dict() )
        return data

    @classmethod
    def from_dict(cls, d:dict, asset:Asset)->Plan:
        """very jank to pass in asset - can definitely do serialisation better"""
        plan = cls(name=d["name"])
        plan.set_asset(asset)
        for v in d["model"]:
            if not v:
                continue
            plan.model.append(Task.from_dict(v, plan=plan))
        return plan





