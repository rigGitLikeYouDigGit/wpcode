
from __future__ import annotations
import typing as T
from pathlib import Path

import hou
from wph import reloadWPH
if T.TYPE_CHECKING:
	from wp.object.filewatcher import ThreadedFileWatcher

nodeWatcherMap : dict[hou.Node, ThreadedFileWatcher] = {}

def fileWatcherForNode(node:hou.Node)->ThreadedFileWatcher:
	"""get file watcher for node, or None"""
	from wp.object.filewatcher import ThreadedFileWatcher
	#print(f"fileWatcherForNode {node.path()}")
	found = nodeWatcherMap.get(node, None)
	#print(f"found {found}")
	return nodeWatcherMap.get(node, None)

def makeNewWatcher(node:hou.Node, kwargs:dict):
	"""make new file observer object, watching all paths
	set on node attributes.

	Call this anytime anything on watcher node changes - safer that way

	can you just make dynamic attrs on hou nodes? do they stick around?
	"""
	# print("")
	# print(f"makeNewWatcher {node.path()}")
	from wp.object.filewatcher import ThreadedFileWatcher

	#print("node map", nodeWatcherMap)
	if node in nodeWatcherMap:
		nodeWatcherMap[node].stop()
		watcher = nodeWatcherMap.pop(node)
		del watcher

	nPaths = node.evalParm("pathfolder")
	paths = []
	buttonParmPaths = []
	for i in range(nPaths):
		paths.append( Path(node.evalParm(f"path{i+1}")).parent )
		buttonParmPaths.append(f"buttonparm{i+1}")

	# print("paths", paths)
	# print("buttonParms", buttonParmPaths)

	def pressButtons():
		"""press all given buttons, but only once each"""
		pressed = set()
		for parmPath in buttonParmPaths:
			parm = node.parm(node.parm(parmPath).eval())
			#print("parm", parm)
			if parm is None:
				print(f"no parm {parmPath} on {node.path()}")
				continue
			if parm in pressed:
				print(f"already pressed {parmPath} on {node.path()}")
				continue
			parm.pressButton()
			#print("pressed button")
			pressed.add(parm)

	watcher = ThreadedFileWatcher(paths)
	def fileEvent(event):
		#print(f"file event {event}")
		pressButtons()
	watcher.setFileEventCallback(fileEvent)

	nodeWatcherMap[node] = watcher
	#print("final node map", nodeWatcherMap)
	return watcher


# node events
def onNodeCreated(node:hou.Node, kwargs):
	"""node created callback"""
	#print("onNodeCreated", node.path())
	makeNewWatcher(node, kwargs)
	setNodeTrackerActive(node, kwargs, True)

def onNodeLoaded(node:hou.Node, kwargs):
	"""node loaded callback"""
	#print("onNodeLoaded", node.path())
	makeNewWatcher(node, kwargs)
	setNodeTrackerActive(node, kwargs, True)

def onUpdated(node:hou.Node, kwargs):
	"""node updated callback"""
	#print("onUpdated", node.path())
	makeNewWatcher(node, kwargs)
	setNodeTrackerActive(node, kwargs, True)

def setNodeTrackerActive(node:hou.Node, kwargs, state=True):
	"""node tracker active callback"""
	#print("setNodeTrackerActive", node.path(), state)
	watcher = fileWatcherForNode(node)
	if watcher:
		if state:
			nodeWatcherMap[node].start()
		else:
			nodeWatcherMap[node].stop()
	else:
		if state:
			makeNewWatcher(node, kwargs).start()




def onNodeDeleted(node:hou.Node, kwargs):
	"""node deleted callback"""
	#print("onNodeDeleted", node.path())
	if node in nodeWatcherMap:
		nodeWatcherMap[node].stop()
		nodeWatcherMap.pop(node)






