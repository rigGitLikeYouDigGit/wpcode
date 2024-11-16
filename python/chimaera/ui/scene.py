
from __future__ import annotations

from chimaera import ChimaeraNode

from wpui.widget.canvas import *
from wpdex import *
from wpdex import react, WX

from .node import NodeDelegate

if T.TYPE_CHECKING:
	pass

class ChimaeraScene(WpCanvasScene):
	"""either ref() into graph data,
	or explicit function to add specific
	node -> get a delegate for it, add to scene, return
	delegate
	"""

	def __init__(self, graph:ChimaeraNode=None,
	             parent=None):
		super().__init__(parent=parent)

		self._graph : ChimaeraNode = WX(None)
		if graph:
			self.setGraph(graph)

	def graph(self)->ChimaeraNode:
		return self._graph.rx.value
	def rxGraph(self)->WX:
		return self._graph
	def setGraph(self, val:ChimaeraNode):
		if react.getRx(val):
			self._graph = val
			self._graph.ref().rx.watch(self._onGraphChanged, onlychanged=False)
		else:
			self._graph.rx.value = val

		self.px = WpDexProxy(EVAL(self._graph))
		self.px.dex().getEventSignal("main").connect(self._onGraphChanged)
		#self._graph.rx.value = self._graph.rx.value
		#self._graph.data.dex().getEventSignal("main").connect(self._onGraphChanged)
		#self.sync() # build out delegates


	def _onGraphChanged(self, *args, **kwargs):
		"""do inefficient check over all nodes in graph,

		TODO: obviously filter to only elements affected by delta
			conform once we have a reasonable syntax for delta events overall

		within event slots like this, we obviously need to go querying
		the rest of the system - changing any data in here may lead to
		infinite loops, be vigilant.

		Also need to work out proper delta comparison, as otherwise it DEFINITELY
		leads to infinite loops
		"""

		log("SCENE on graph changed", args, kwargs)
		currentDelegates = set(i for i in self.items() if isinstance(i, WpCanvasElement))

		# TODO: later a single node may create multiple delegates - group boxes, ports etc
		#   for now assume one to one

		# remove
		nodesToMatch = set(self.graph().branches)
		for i in currentDelegates:
			if i.obj in nodesToMatch:
				nodesToMatch.remove(i.obj)
			if not i.obj in nodesToMatch:
				self.removeItem(i)
				currentDelegates.remove(i)


		for node in nodesToMatch:
			#log("get delegate for", node, type(node))
			delegateType = NodeDelegate.adaptorForObject(node)
			newDel = self.addItem(delegateType(node))
			#TODO: support node delegate creating its own secondary elements





	def sync(self, elements=()):
		if not elements:
			self.clear()
		for name, node in self.graph().branchMap(): pass


	pass


