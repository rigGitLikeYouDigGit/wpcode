
from __future__ import annotations
import typing as T

from pathlib import Path

from wplib import sequence

import bpy

from bpy.utils import script_path_user

def getRecentFiles()->list[Path]:
	"""returns a list of recently opened files"""

	fp = bpy.utils.user_resource('CONFIG', path="recent-files.txt")
	print("fp", fp)
	try:
		with open(fp) as file:
			recent_files = file.read().splitlines()
	except (IOError, OSError, FileNotFoundError):
		recent_files = []
	if not recent_files:
		return []
	return [Path(i) for i in recent_files]


def deleteAddOn(addOnPath:Path):
	"""deletes the add-on file in the blender
	plugin dir matching name of given module.

	Prevents infuriating and inconsistent copying/caching
	python files

	"""
	addonModuleName = addOnPath.name
	addonDir = Path(script_path_user()) / "addons"
	addonFile = addonDir / addonModuleName
	if addonFile.exists():
		addonFile.unlink()


def registerAddOn(addOnPath:Path):
	"""registers the add-on at the given path"""
	deleteAddOn(addOnPath)

	installResult = bpy.ops.preferences.addon_install(filepath=addOnPath.as_posix())
	print("installResult", installResult)
	enableResult = bpy.ops.preferences.addon_enable(module=addOnPath.stem)
	print("enableResult", enableResult)


def listDescendents(obj:bpy.types.Object, l:list)->list[bpy.types.Object]:
	"""returns a list of all descendents of the given object,
	including original object"""
	l.append(obj)
	for child in obj.children:
		listDescendents(child, l)
	return l


def selection(context=None)->list[bpy.types.Object]:
	"""returns a list of all selected objects"""
	context = context or bpy.context
	print("selection", context.selected_objects, type(context.selected_objects))
	return context.selected_objects


def select(*obj:(bpy.types.Object, T.Sequence[bpy.types.Object]), state=True):
	"""selects the given object"""
	obj = sequence.flatten(obj)
	for i in obj:
		i.select_set(state)


def clearSelection():
	"""deselects all objects"""
	bpy.ops.object.select_all(action='DESELECT')
	#
	# for i in selection():
	# 	select(i, False)