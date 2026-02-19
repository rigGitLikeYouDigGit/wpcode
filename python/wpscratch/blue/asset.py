from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from __future__ import annotations
import importlib, traceback
from pathlib import Path
import typing as T, types
from dataclasses import dataclass

@dataclass
class Asset:
    show : str
    type : str
    asset : str
    rig : str = "rig"


    def showDir(self)->Path:
        return Path("/jobs") / self.show
    def assetDir(self)->Path:
        return self.showDir() / "build" / self.type / self.asset
    def scriptDir(self)->Path:
        return self.assetDir() / ("m_" + self.asset) / "scripts"
        #self.path = '/jobs/{0}/build/{1}/{2}/m_{2}/scripts/'.format(show, asset_type, asset)

    def tokens(self)->list[str]:
        return [self.show, self.type, self.asset]

    def exists(self):
        #print("exists?", self.scriptDir(), self.scriptDir().exists(), self.scriptDir().is_dir())
        return self.scriptDir().is_dir()

    def rigPathMap(self)->dict[str, Path]:
        # print(self.showDir())
        # print(self.assetDir())
        # print(self.scriptDir())
        pathMap = {}
        for i in self.assetDir().iterdir():
            if(str(i).endswith("plan")):
                pathMap[i.stem] = i
        return pathMap

    @staticmethod
    def jobsDir()->Path:
        return Path("/jobs")

    @classmethod
    def showDirMap(cls)->dict[str, Path]:
        return {i.stem : i for i in cls.jobsDir().iterdir() if i.is_dir()}

    @classmethod
    def assetTypeDirMap(cls, showDir:Path)->dict[str, Path]:
        return {i.stem : i for i in (showDir / "build").iterdir() if i.is_dir()}

    @classmethod
    def assetDirMap(cls, showAssetTypeDir:Path)->dict[str, Path]:
        return {i.stem : i for i in (showAssetTypeDir).iterdir() if i.is_dir()}



# asset_typ e ='character', mod e =None, build_class_typ e ='rig_build', build_class_nam e ='RigBuild'):
# '''
# Args:
# asset (string) - asset name.
# show (string) - job name.
# asset_type (string) - character vehicle prop, defaults = character.
# '''
#
# self.logger = logging.getLogger(self.__class__.__name__)
# self.logger.setLevel(logging.INFO)
#
# self.asset = asset
# self.show = show
# self.asset_type = asset_type
# self.mode = mode
# self.build_class_type = build_class_type
# self.build_class_name = build_class_name
#
# self.build_complete = False
#
# self.path = '/jobs/{0}/build/{1}/{2}/m_{2}/scripts/'.format(show, asset_type, asset)