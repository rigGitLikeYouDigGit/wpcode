
from __future__ import annotations
import typing as T

from pxr import Usd

import bpy

from pathlib import Path
from tree.lib.object import TypeNamespace


"""create new io-collection with given name
each has import and export path, and import and export button"""

"""
bpy.ops.wm.path_open(filepath='')
Open a path in a file browser

Parameters
filepath (string, (optional, never None)) – filepath

File
startup/bl_operators/wm.py:1122

"""

class EmptyDrawMode(TypeNamespace):

	class _Base(TypeNamespace.base()):
		s = ""
		pass

	class PlainAxes(_Base):
		s = "PLAIN_AXES"
		pass

def makeNewEmpty(name:str)->bpy.types.Object:
	o = bpy.data.objects.new("empty", None)
	bpy.context.scene.collection.objects.link(o)
	o.empty_display_type = EmptyDrawMode.PlainAxes.s
	return o

def deleteCollectionAndChildren(name:str):
	"""deletes the collection with the given name,
	also removing all items in it"""
	collection = bpy.data.collections[name]
	# remove all children
	for child in collection.objects:
		bpy.data.objects.remove(child, do_unlink=True)
	# remove collection
	bpy.data.collections.remove(collection)

def listDescendents(obj:bpy.types.Object, l:list)->list[bpy.types.Object]:
	"""returns a list of all descendents of the given object,
	including original object"""
	l.append(obj)
	for child in obj.children:
		listDescendents(child, l)
	return l

def selection()->list[bpy.types.Object]:
	"""returns a list of all selected objects"""
	return bpy.context.selected_objects

def select(obj:(bpy.types.Object, T.Sequence[bpy.types.Object]), state=True):
	"""selects the given object"""
	if not isinstance(obj, (tuple, list, set)):
		obj = [obj]
	for i in obj:
		i.select_set(state)

def clearSelection():
	"""deselects all objects"""
	for i in selection():
		select(i, False)

def makeIoNull(name:str):
	"""creates a new io null with the given name,
	allowing consistent import and export of usd files"""
	# for now delete null if already exists
	print("name", name, name in bpy.data.objects)
	if name in bpy.data.objects:
		for i in listDescendents(bpy.data.objects[name], []):
			bpy.data.objects.remove(i, do_unlink=True)
	# create null
	topNull = makeNewEmpty(name)
	print(topNull, type(topNull))
	topNull.name = name
	# ioColl = bpy.data.collections.new(name)
	# # add to scene
	# bpy.context.scene.collection.children.link(ioColl)
	# # create import and export path properties
	defaultInPath = "../in/sculpt_in.usda"
	importPathProp = bpy.props.StringProperty(
		name="importPath", default=defaultInPath,
		description="path to usd file to import from",
		#subtype="FILE_PATH"
	)
	#topNull["importPath"] = importPathProp
	topNull["importPath"] = defaultInPath

EXPORT_PATH_PROP_NAME = "exportPath"

def exportUsdFromNull(null:bpy.types.Object, toPath:Path=None):
	"""if path not specified, look up from exportPath property"""
	if toPath is None:
		toPath = Path(null[EXPORT_PATH_PROP_NAME])
	#print("toPath", toPath, toPath.is_absolute())
	if not toPath.is_absolute():
		scenePath = Path(bpy.data.filepath)
		toPath = scenePath.parent / toPath

	# check we're exporting to usda
	if not toPath.suffix == ".usda":
		toPath = (toPath.parent / toPath.stem).with_suffix(".usda")

	nodes = listDescendents(null, [])

	baseSelection = selection()
	clearSelection()
	select(nodes)

	# export to path - blender doesn't expose much to USD, it's all or nothing for now
	bpy.ops.wm.usd_export(
		filepath=str(toPath),
		check_existing=False,
		selected_objects_only=True,
		#apply_modifiers=True,
		visible_objects_only=False,
		export_materials=False, # skip materials for now
		generate_preview_surface=False,
		export_textures=False,
	)

	# restore selection
	clearSelection()
	select(baseSelection)





