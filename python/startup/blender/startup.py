
from __future__ import annotations

import typing as T

import sys, os, shutil, json, inspect
from pathlib import Path
"""script run on blender startup"""

# add wpblend to path
wpPath = Path(__file__).parents[3]

if not wpPath.as_posix() in sys.path:
	sys.path.append(wpPath.as_posix())

# os.environ["BLENDER_VAR"] = "adkjhkjakfagaskjd"
#print("check from launcher", os.environ["BLENDER_VAR"])

import bpy
import wpblend
from wpblend import lib

#print("wpblend", wpblend)

# register plugins
addonDir = Path(wpblend.__file__).parent / "addon"
addonLoadOrder = json.load((addonDir / "loadOrder.json").open("r"))
#print("addonLoadOrder", addonLoadOrder)

for i in addonLoadOrder:
	print("loading", i)
	filePath = addonDir / i
	lib.registerAddOn(filePath)


# try to reopen last file
recentFiles = lib.getRecentFiles()
if recentFiles:
	print("recentFiles", recentFiles)
	filePath = recentFiles[0]
	bpy.ops.wm.open_mainfile(filepath=filePath.as_posix())


print("\nwpblend startup complete\n\n\n")
