
from __future__ import annotations

import bpy

from pathlib import Path
from tree.lib.object import TypeNamespace
from wpblend.lib import listDescendents, selection, select, clearSelection

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
IMPORT_PATH_PROP_NAME = "importPath"


def exportUsdFromNull(null:bpy.types.Object, toPath:Path=None):
	"""if path not specified, look up from exportPath property"""
	if toPath is None:
		try:
			toPath = Path(null[EXPORT_PATH_PROP_NAME])
		except KeyError as k:
			k.args = (*k.args, f"must supply either explicit export path or set {EXPORT_PATH_PROP_NAME} property on null f{null}")
			raise k
	#print("toPath", toPath, toPath.is_absolute())
	if not toPath.is_absolute():
		scenePath = Path(bpy.data.filepath)
		toPath = scenePath.parent / toPath
		if toPath.resolve() == scenePath.resolve():
			raise ValueError(f"export path {toPath} is same as scene path {scenePath}")

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


def importUsdToNull( null:bpy.types.Object, fromPath:Path=None ):
	"""if path not specified, look up from importPath property"""
	if fromPath is None:
		try:
			fromPath = Path(null[IMPORT_PATH_PROP_NAME])
		except KeyError:
			raise KeyError(f"must supply either explicit import path or set {IMPORT_PATH_PROP_NAME} property on null")
	#print("fromPath", fromPath, fromPath.is_absolute())
	if not fromPath.is_absolute():
		scenePath = Path(bpy.data.filepath)
		fromPath = scenePath.parent / fromPath

	# check we're importing from usda
	if not fromPath.suffix == ".usda":
		fromPath = (fromPath.parent / fromPath.stem).with_suffix(".usda")

	# import from path
	bpy.ops.wm.usd_import(
		filepath=str(fromPath)
	)

