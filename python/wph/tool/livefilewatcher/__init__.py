
import hou

from wp.object.filewatcher import ThreadedFileWatcher

def makeNewWatcher(node:hou.Node, kwargs:dict):
	"""make new file observer object, watching all paths
	set on node attributes.

	Call this anytime anything on watcher node changes - safer that way

	can you just make dynamic attrs on hou nodes? do they stick around?
	"""

	if node._fileWatcher

	nPaths = node.evalParm("pathfolder")
	paths = []
	buttonParms = []
	for i in range(nPaths):
		paths.append( node.evalParm(f"path{i+1}") )
		buttonParms.append(f"buttonparm{i+1}")

	def pressButtons():
		"""press all given buttons, but only once each"""
		pressed = set()
		for parmPath in buttonParms:
			parm = node.parm(parmPath)
			if parm is None:
				print(f"no parm {parmPath} on {node.path()}")
				continue
			if parm in pressed:
				continue
			node.parm(i).pressButton()
			pressed.add(parm)

	watcher = ThreadedFileWatcher(paths)
	watcher.onFileEvent = lambda event : pressButtons()
	return watcher






