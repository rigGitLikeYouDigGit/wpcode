
from __future__ import annotations
import typing as T

from wplib import inheritance

class PluginBase:
	"""Base class for plugins -
	for each new register system, define a plugin base.
	This base should define all interface methods the
	system can look for -
	then each subsequent real plugin can override them
	"""

	@classmethod
	def checkIsValid(cls)->bool:
		return True





class PluginRegister:
	"""initialise a new register object for each system -
	keep logic for plugin lookup simple"""

	def __init__(self,
	             #pluginBaseCls,
	             systemName:str):
		#self.pluginBaseCls = pluginBaseCls
		self.pluginMap : dict[T.Any, T.Type[PluginBase]] = {}
		self.name = systemName


	def registerPlugin(self, plugin:T.Type[PluginBase], key:(str, type, tuple[type])):
		"""Register against a string key or a set of types -
		a base class lookup is done to retrieve them"""
		assert plugin.checkIsValid(), f"Plugin {plugin} is not valid"
		assert key not in self.pluginMap, f"Key {key} already registered in {self} plugin map:\n{self.pluginMap}"
		self.pluginMap[key] = plugin

	def getPlugin(self, key:(str, type, tuple[type]))->T.Type[PluginBase]:
		try:
			return self.pluginMap[key]
		except KeyError:
			pass
		return inheritance.superClassLookup(self.pluginMap, key, None)
