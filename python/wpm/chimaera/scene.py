
from __future__ import annotations
import typing as T

from pathlib import Path

from wplib import log

from wpexp.plugin import SceneExpPlugin, ExpError, ExpWarning

from wpm import cmds, om
"""integration with maya scene, file paths, etc
name of top chimaera node is the name of the graph file?



open up for the first time in a scene, outside of any asset system - 

default chimaera graph path is $SCENE_DIR/chimaera/

if maya scene has not yet been saved, scene_dir expression fails with a visible warning,
getting save path raises an error

FOR NOW, save everything under top node in a single file, including data

"""


class MayaSceneExpPlugin(SceneExpPlugin):

	def resolveToken(self, token:str) ->str:
		"""resolve a token to a value"""
		if token in ("SCENE_DIR", "SCENE_NAME"):
			scenePath = cmds.file(q=True, sceneName=True)
			log("found scene path", scenePath)
			if not scenePath:
				raise ExpWarning(token, "maya scene has not been saved")
			if token == "SCENE_DIR":
				return Path(scenePath).parent
			elif token == "SCENE_NAME":
				return Path(scenePath).stem