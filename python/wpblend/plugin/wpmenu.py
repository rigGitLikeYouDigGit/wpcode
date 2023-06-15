
from __future__ import annotations

import typing as T

import bpy

"""add extensible menu to blender at startup, so we can 
easily add more stuff to it"""


class ExtendMenu(bpy.types.Menu):
	"""need to subclass for each new kind of menu sadly"""

	bl_idname = "wp.extend_menu"
	bl_label = "Extend"

	# map of {idname: menu label}
	_itemMap : T.Dict[str, str] = {}

	def draw(self, context):
		layout = self.layout
		for k, v in self._itemMap.items():
			layout.operator(k, text=v)


class WPMainMenu(ExtendMenu):

	bl_idname = "wp.wp_main_menu"
	bl_label = "WP"
	_itemMap = {}

