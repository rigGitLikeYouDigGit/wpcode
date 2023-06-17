from __future__ import annotations

import typing as T

import sys, os, shutil, json
from importlib import reload
from pathlib import Path
import bpy
from wpblend import assetio, lib

#from wpblend.addon.wpmenu import WP_MT_main_menu

"""operators integrating Blender with WP io system"""

bl_info = {
	"name": "WP_io",
	"blender": (2, 80, 0),
	"category": "Object",
}



class SyncIONull(bpy.types.Operator):
	"""Update import and export content under selected nulls"""
	# Use this as a tooltip for menu items and buttons.
	bl_idname = "object.wp_sync_io" # can't have wrong casing in here, or it's an 'invalid_idname'
	bl_label = "WP sync selected IO nulls"         # Display name in the interface.
	bl_options = {'REGISTER', # 'UNDO' # YOU CAN'T UNDO THIS
	              }

	def execute(self, context):        # execute() is called when running the operator.
		"""lib.selection() doesn't pick up selected objects properly
		when called from within operator, even as
		the exact same code works fine in the console."""

		null = None

		if lib.selection(context):
			null = lib.selection(context)[0]
		elif context.active_object:
			null = context.active_object
		elif context.object:
			null = context.object

		if null is None:
			raise Exception("No null selected")

		assetio.exportUsdFromNull(null)

		return {'FINISHED'} # WHY
		# do you return a SET
		# containing A SINGLE STRING

def menu_func(self, context):
	self.layout.operator(SyncIONull.bl_idname)

def register():
	bpy.utils.register_class(SyncIONull)
	#bpy.types.VIEW3D_MT_object.append(menu_func)  # Adds the new operator to an existing menu.
	#WP_MT_main_menu.append(menu_func)  # Adds the new operator to an existing menu.

def unregister():
	bpy.utils.unregister_class(SyncIONull)


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
	register()
