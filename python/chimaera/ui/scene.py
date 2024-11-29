
from __future__ import annotations

from chimaera import ChimaeraNode

from wpui.canvas import *
from wplib.inheritance import resolveInheritedMetaClass
from wpdex import *
from wpdex import WX
from wpdex.ui import AtomicUiInterface
from .node import NodeDelegate

if T.TYPE_CHECKING:
	pass

class ChimaeraScene(
	WpCanvasScene,
	AtomicUiInterface,
	metaclass=resolveInheritedMetaClass(
		WpCanvasScene, AtomicUiInterface
	)
                    ):
	"""either ref() into graph data,
	or explicit function to add specific
	node -> get a delegate for it, add to scene, return
	delegate
	"""

	def __init__(self, graph:ChimaeraNode=None,
	             parent=None):
		AtomicUiInterface.__init__(self,
		                           value=graph)
		WpCanvasScene.__init__(self, parent=parent)


	def graph(self)->ChimaeraNode | WpDexProxy:
		return self.valueProxy()

	def rawGraph(self)->ChimaeraNode:
		return self.value()

	def rxGraph(self)->WX:
		return self.rxValue()

	def _tryCommitValue(self, value):
		"""everything managed by internal node signals,
		graph view itself is straight display"""
		pass

	def _syncUiFromValue(self, *args, **kwargs):
		"""do inefficient check over all nodes in graph,

		TODO: obviously filter to only elements affected by delta
			conform once we have a reasonable syntax for delta events overall

		within event slots like this, we obviously need to go querying
		the rest of the system - changing any data in here may lead to
		infinite loops, be vigilant.

		Also need to work out proper delta comparison, as otherwise it DEFINITELY
		leads to infinite loops
		"""
 
		log("SCENE sync ui", args, kwargs)
		#currentDelegates = set(i for i in self.items() if isinstance(i, WpCanvasElement))
		currentDelegates = set(i for i in self.items() if isinstance(i, NodeDelegate))
  
		# TODO: later a single node may create multiple delegates - group boxes, ports etc
		#   for now assume one to one

		# remove
		nodesToMatch = set(self.graph().branches)
		for i in tuple(currentDelegates):
			if i.obj in nodesToMatch:
				nodesToMatch.remove(i.obj)
			else: # dangling delegate, remove
				self.removeItem(i)
				currentDelegates.remove(i)


		for node in nodesToMatch:
			#log("get delegate for", node, type(node))
			delegateType = NodeDelegate.adaptorForObject(node)
			newDel = self.addItem(delegateType(node))
			#TODO: support node delegate creating its own secondary elements

		endDelegates = set(i for i in self.items() if isinstance(i, WpCanvasElement))
		# log("end items", self.items())
		# for i in self.items():
		# 	log(isinstance(i, WpCanvasElement), type(i), type(i).__mro__, i)
		log("endDelegates", endDelegates)





	pass

