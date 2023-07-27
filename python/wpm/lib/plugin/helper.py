
from __future__ import annotations
import typing as T

import sys, os, traceback

from pathlib import Path

from maya.api import OpenMaya as om, OpenMayaRender as omr
from maya import cmds

from wpm.constant import WP_PLUGIN_NAMESPACE, PY_PLUGIN_START_ID
from wpm.lib.plugin.template import PluginNodeTemplate, PluginDrawOverrideTemplate, pluginNodeType

pluginDrawOverrideType = (PluginDrawOverrideTemplate, omr.MPxDrawOverride)
class MayaPluginAid:
	"""helper object for defining boilerplate around maya plugins - registering nodes, linking draw overrides, etc

	should be relatively robust to reloading - unloadPlugin() here
	should catch a plugin of the same name even if all the related python files have been reloaded since it was initialised in maya

	ALSO handles handing out node type ids - each individual plugin gets its own
	range of 16 ids - it's enough for now

	ids can really mess stuff up if they ever change incorrectly - for now we do
	asserts around them as the first operation, but really we need a more robust
	solution
	"""

	registeredPlugins : dict[str, MayaPluginAid] = {}  # dict of { plugin name : [ registered plugin classes ] }

	pluginIdInstanceMap : dict[int, MayaPluginAid] = {}


	def __init__(self, name:str,
	             studioLocalPluginId:int,
	             pluginPath:(str, Path),
	             nodeClasses: tuple[pluginNodeType] = (),
	             drawOverrideClasses: dict[pluginNodeType,
	                                      pluginDrawOverrideType] = {},
	             useOldApi:bool = False
	             ):
		"""
		:param name: string name of plugin
		:param studioLocalPluginId: local id of this overall plugin, used for
			building node type ids - must be between 0 and 12
		:param pluginPath: file path to this plugin's main file
		:param nodeClasses: tuple of node classes to register with this plugin
		:param drawOverrideClasses: dict of [ nodeClass : drawOverride for that class ]
		"""
		self.name = name
		self.studioLocalPluginId = studioLocalPluginId

		self.pluginPath : Path = Path(pluginPath)
		self.nodeClasses = nodeClasses
		self.drawOverrideClasses = drawOverrideClasses

		self.useOldApi = useOldApi

		self._mfnPlugin : om.MFnPlugin = None

	def nodeClsTypeId(self, cls:T.Type[pluginNodeType])->om.MTypeId:
		if self.useOldApi:
			from maya import OpenMaya as omOld
			return omOld.MTypeId(PY_PLUGIN_START_ID + 16 * self.studioLocalPluginId + cls.kNodeLocalId)
		return om.MTypeId(WP_PLUGIN_NAMESPACE,
		                  PY_PLUGIN_START_ID + 16 * self.studioLocalPluginId + cls.kNodeLocalId)

	def _registerNode(self, cls: T.Type[pluginNodeType],
	                  ):
		assert cls.kNodeLocalId > -1, "node {} must define its own plugin-local id".format(cls)
		nodeId = self.nodeClsTypeId(cls)
		try:
			if cls.kDrawClassification:
				self._mfnPlugin.registerNode(cls.kNodeName, nodeId, lambda: cls(), cls.initialiseNode,
				                    cls.kNodeType, cls.kDrawClassification
				                    )
			else:
				self._mfnPlugin.registerNode(cls.kNodeName, nodeId, lambda: cls(), cls.initialiseNode,
				                    cls.kNodeType
				                    )
		except:
			print('Failed to register node: ' + cls.kNodeName)
			traceback.print_exc()

	def _deregisterNode(self, cls: T.Type[pluginNodeType]):
		nodeId = self.nodeClsTypeId(cls)
		try:
			self._mfnPlugin.deregisterNode(nodeId)
		except:
			print('Failed to deregister node: ' + cls.kNodeName)
			traceback.print_exc()

	def _registerDrawOverride(self, drawOverrideCls: T.Type[pluginDrawOverrideType],
	                          forNodeCls: T.Type[pluginNodeType]):
		"""register a drawOverride for a specific node class"""
		assert forNodeCls.kDrawClassification, "Node to be drawn must define a draw classification \nsuch as 'drawdb/geometry/<nodeName>"
		try:
			omr.MDrawRegistry.registerDrawOverrideCreator(
				forNodeCls.kDrawClassification,
				drawOverrideCls.kDrawRegistrantId(),
				drawOverrideCls.creator
			)
		except Exception as e:
			print("failed to register draw override {} for node class {}".format(drawOverrideCls, forNodeCls))
			traceback.print_exc()

	def _deregisterDrawOverride(self, drawOverrideCls: T.Type[pluginDrawOverrideType],
	                          forNodeCls: T.Type[pluginNodeType]):
		try:
			omr.MDrawRegistry.deregisterDrawOverrideCreator(
				forNodeCls.kDrawClassification,
				drawOverrideCls.kDrawRegistrantId()
			)
		except Exception as e:
			print("failed to deregister draw override {} for node class {}".format(drawOverrideCls, forNodeCls))
			traceback.print_exc()

	def initialisePlugin(self, pluginMObject:om.MObject):
		"""call this from within the initializePlugin() function in your main plugin file.
		:param pluginMObject: the MObject passed into initializePlugin()
		"""
		self._mfnPlugin = om.MFnPlugin(pluginMObject)

		try:
			# add this plugin to global register
			assert self.studioLocalPluginId not in self.pluginIdInstanceMap, f"{self} tried to register duplicate local plugin index {self.studioLocalPluginId}\n existing indices: {self.pluginIdInstanceMap} "
			self.pluginIdInstanceMap[self.studioLocalPluginId] = self

			# register nodes
			for i in self.nodeClasses:
				self._registerNode(i)

			# register draw overrides
			for nodeCls, drawOverrideCls in self.drawOverrideClasses.items():
				self._registerDrawOverride(drawOverrideCls, forNodeCls=nodeCls)
		except Exception as e:
			print(f"unable to initialise plugin {self}")
			traceback.print_exc()

	def uninitialisePlugin(self, pluginMObject:om.MObject):
		"""call this from within the uninitializePlugin() function in your main plugin file
				"""
		self._mfnPlugin = om.MFnPlugin(pluginMObject)
		# deregister plugin items in reverse order to registration
		for nodeCls, drawOverrideCls in self.drawOverrideClasses.items():
			self._deregisterDrawOverride(drawOverrideCls, forNodeCls=nodeCls)

		for i in self.nodeClasses:
			self._deregisterNode(i)

	def initialisePluginOldApi(self, pluginMObject:om.MObject):
		import maya.OpenMaya as omOld
		import maya.OpenMayaMPx as ompx
		self._mfnPlugin = ompx.MFnPlugin(pluginMObject)
		for i in self.nodeClasses:
			assert i.kNodeLocalId > -1, "node {} must define its own plugin-local id".format(i)
			nodeTypeId = self.nodeClsTypeId(i)
			self._mfnPlugin.registerNode(i.kNodeName, nodeTypeId, i.nodeCreator, i.initialiseNode, i.kNodeType)

	def uninitialisePluginOldApi(self, pluginMObject:om.MObject):
		""""""
		import maya.OpenMaya as omOld
		import maya.OpenMayaMPx as ompx
		self._mfnPlugin = ompx.MFnPlugin(pluginMObject)
		for i in self.nodeClasses:
			nodeTypeId = self.nodeClsTypeId(i)
			self._mfnPlugin.deregisterNode(nodeTypeId)

	def isRegistered(self)->bool:
		"""check if plugin represented by this object is currently registered in maya"""
		return cmds.pluginInfo(self.name, q=1, registered=1)

	def isLoaded(self)->bool:
		"""check if plugin represented by this object is currently loaded in maya"""
		try:
			return cmds.pluginInfo(self.name, q=1, loaded=1)
		except RuntimeError: # if plugin is not registered
			return False

	def loadPlugin(self, forceReload=True):
		"""if forceReload, will unload plugin if currently loaded
		"""
		if self.isRegistered():
			if self.isLoaded():
				if forceReload:
					self.unloadPlugin()

		# check that file lies on maya plugin path - else add it
		pluginDirs = os.getenv('MAYA_PLUG_IN_PATH').split(';')
		if not self.pluginPath in pluginDirs:
			pluginDirs.append(str(self.pluginPath))
			os.putenv("MAYA_PLUG_IN_PATH", ';'.join(pluginDirs))

		cmds.loadPlugin(str(self.pluginPath), name=self.name)
		assert self.isLoaded(), "loading plugin {} did not properly load it in Maya".format((self.name, self.pluginPath, self))

	def unloadPlugin(self):
		if self.isLoaded():
			cmds.unloadPlugin(self.name, force=1)
		assert not self.isLoaded(), "unloading plugin {} did not properly unload it in Maya".format((self.name, self.pluginPath, self))



