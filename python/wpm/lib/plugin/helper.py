
from __future__ import annotations

import pprint
import typing as T

import sys, os, traceback

from pathlib import Path

from wplib import log

from maya.api import OpenMaya as om, OpenMayaRender as omr
from maya import cmds

from wpm.constant import WP_PLUGIN_NAMESPACE, PY_PLUGIN_START_ID
from wpm.lib.plugin.template import PluginNodeTemplate, PluginDrawOverrideTemplate, pluginNodeType

pluginDrawOverrideType = (PluginDrawOverrideTemplate, omr.MPxDrawOverride)
class MayaPluginAid:
	"""helper object for defining boilerplate around maya plugins - registering nodes, linking draw overrides, etc

	should be relatively robust to reloading - unloadPlugin() here
	should catch a plugin of the same name even if all the related python files have been reloaded since it was initialised in maya

	studio id prefix allows 256 total nodes in a studio's plugin namespace
	register nodes with an id between 0 and 255

	IDs are managed ENTIRELY from this aid object - node wrappers do not
	define them
	"""

	#registeredPlugins : dict[str, MayaPluginAid] = {}  # dict of { plugin name : [ registered plugin classes ] }

	#pluginIdInstanceMap : dict[int, MayaPluginAid] = {}

	@classmethod
	def studioIdPrefix(cls)->int:
		"""OVERRIDE
		return the prefix for studio plugin ids,
		used as a namespace for plugin node ids"""
		raise NotImplementedError


	def __init__(self, name:str,
	             #studioLocalPluginId:int,
	             pluginPath:(str, Path),
	             nodeClasses: dict[int, type[PluginNodeTemplate]] = {},
	             drawOverrideClasses: dict[pluginNodeType,
	                                      pluginDrawOverrideType] = {},
	             useOldApi:bool = False
	             ):
		"""
		:param name: string name of plugin

		:param pluginPath: file path to this plugin's main file -
			should usually be the same file that initialises a PluginAid object,
			with separate initializePlugin() and uninitializedPlugin() functions at module level

		:param nodeClasses: tuple of node classes to register with this plugin
		:param drawOverrideClasses: dict of [ nodeClass : drawOverride for that class ]
		"""
		self.name = name
		#self.studioLocalPluginId = studioLocalPluginId

		self.pluginPath : Path = Path(pluginPath)
		self.nodeClasses = dict(nodeClasses)
		self.drawOverrideClasses = dict(drawOverrideClasses)

		self.useOldApi = useOldApi

		self._mfnPlugin : om.MFnPlugin = None



	def nodeClsTypeId(self, cls:T.Type[PluginNodeTemplate], nodeId:int)->om.MTypeId:


		if self.useOldApi:
			from maya import OpenMaya as omOld
			return omOld.MTypeId(WP_PLUGIN_NAMESPACE, nodeId)
		return om.MTypeId(WP_PLUGIN_NAMESPACE, nodeId)

	def _registerNode(self, cls: T.Type[PluginNodeTemplate],
	                  localNodeId:int):
		assert 0 < localNodeId and localNodeId < 256, "node {} must define its own plugin-local id between 0 and 256".format(cls)
		nodeId = self.nodeClsTypeId(cls, localNodeId)
		try:
			if cls.kDrawClassification:
				self._mfnPlugin.registerNode(cls.typeName(), nodeId, lambda: cls(), cls.initialiseNode,
				                    cls.kNodeType(), cls.kDrawClassification
				                    )
			else:
				self._mfnPlugin.registerNode(cls.typeName(), nodeId, lambda: cls(), cls.initialiseNode,
				                    cls.kNodeType()
				                    )
		except:
			print('Failed to register node: ' + cls.kNodeName())
			traceback.print_exc()

	def _deregisterNode(self, cls: T.Type[PluginNodeTemplate]):
		nodeLocalId = {v:k for k,v in self.nodeClasses.items()}[cls]
		nodeId = self.nodeClsTypeId(cls, nodeLocalId)
		try:
			self._mfnPlugin.deregisterNode(nodeId)
		except:
			print('Failed to deregister node: ' + cls.kNodeName())
			traceback.print_exc()

	def _registerDrawOverride(self, drawOverrideCls: T.Type[pluginDrawOverrideType],
	                          forNodeCls: T.Type[PluginNodeTemplate]):
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
	                          forNodeCls: T.Type[PluginNodeTemplate]):
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
			# # add this plugin to global register
			# assert self.studioLocalPluginId not in self.pluginIdInstanceMap, f"{self} tried to register duplicate local plugin index {self.studioLocalPluginId}\n existing indices: {self.pluginIdInstanceMap} "
			# self.pluginIdInstanceMap[self.studioLocalPluginId] = self

			# register nodes
			for localId, nodeCls in self.nodeClasses.items():
				self._registerNode(nodeCls, localId)

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

		for nodeCls in self.nodeClasses.values():
			self._deregisterNode(nodeCls)

	def initialisePluginOldApi(self, pluginMObject:om.MObject):
		import maya.OpenMaya as omOld
		import maya.OpenMayaMPx as ompx
		self._mfnPlugin = ompx.MFnPlugin(pluginMObject)
		for i in self.nodeClasses:
			assert i.kNodeLocalId > -1, "node {} must define its own plugin-local id".format(i)
			nodeTypeId = self.nodeClsTypeId(i)
			self._mfnPlugin.registerNode(i.kNodeName(), nodeTypeId, i.nodeCreator, i.initialiseNode, i.kNodeType())

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
		assert not self.isLoaded(), "unloading plugin {} did not properly unload it in Maya\n good luck lol".format((self.name, self.pluginPath, self))

	def testPlugin(self):
		"""create all nodes, run all node test functions"""
		for nodeCls in self.nodeClasses.values():
			nodeCls.testNode()

	def pluginNodeTypeNames(self):
		"""return type names of all nodes added by plugin"""
		nodeClasses = cmds.pluginInfo(self.name, q=1, dependNode=1) or []
		return nodeClasses

	def updateGeneratedNodeClasses(self):
		"""refresh generated node wrapper classes under
		wpm/core/node/gen
		with all nodes added by this plugin.
		This of course will not touch any classes under node/author
		"""
		import tempfile
		from wpm.core.node._codegen import gather, generate
		outputPath = gather.TARGET_NODE_DATA_PATH.parent / "pluginNodeData.json"
		nodeClassNames = self.pluginNodeTypeNames()
		if not nodeClassNames:
			log("no node classes to update wrappers for plugin " + self.name)
			return
		nodeTypes = gather.gatherNodeData(nodeClassNames, outputPath)
		#log("classes to update:")
		#pprint.pprint(nodeTypes)
		generate.genNodes(jsonPath=outputPath,
		                  refreshGenInitFile=True)
		# self._mfnPlugin = om.MFnPlugin(
		# 	om.MFnPlugin().findPlugin(self.name)
		# )

