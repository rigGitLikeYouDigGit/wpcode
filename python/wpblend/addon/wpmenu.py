
from __future__ import annotations

import typing as T
import inspect
import bpy, bl_ui

from wpblend.addon.wpio import SyncIONull

"""add extensible menu to blender at startup, so we can 
easily add more stuff to it"""

bl_info = {
	"name": "WP_mainMenu",
	"blender": (2, 80, 0),
	"category": "Object",
}



class WP_MT_main_menu(bpy.types.Menu): # class names are enforced by API

	bl_idname = "WP.VIEW3D_MT_wp_main_menu"
	bl_label = "wp"

	@classmethod
	def targetParent(cls):
		return bpy.types.VIEW3D_HT_header

	def draw(self, context):
		layout = self.layout
		layout.operator(SyncIONull.bl_idname, text=SyncIONull.bl_label)

def draw_item(self, context):
	layout = self.layout
	layout.menu(WP_MT_main_menu.bl_idname)

def register():
	bpy.utils.register_class(WP_MT_main_menu)

	WP_MT_main_menu.targetParent().prepend(draw_item)  # Adds the new operator to an existing menu.
	"""
	'why is WP on the very left side of the bar?'
	Because drawing the entire toolbar is handled within the draw() function
	of VIEW3D_HT_header, and changing that would involve patching WP
	into the source code itself. Could do it eventually, but too complex for now.
	
	"""


def unregister():
	bpy.utils.unregister_class(WP_MT_main_menu)
	WP_MT_main_menu.targetParent().remove(draw_item)


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
	register()


