
import pprint

from wplib.object import DirtyGraph, DirtyNode

from wptree import Tree

from chimaera.core.node import ChimaeraNode
#from chimaera.core.graph import ChimaeraGraph
from chimaera.core.exegraph import ChimaeraDirtyGraph
from chimaera.core.plugnode import PlugNode


def makePlugNode():

	class TestPlugNode(PlugNode):

		@classmethod
		def makePlugs(cls, inRoot:Tree, outRoot:Tree) ->None:
			print("makePlugs")
			inRoot("inPlug", create=True)
			outRoot("outPlug", create=True)

	graph = ChimaeraNode()
	node = graph.createNode("topNodeA")

	plugNode : TestPlugNode = graph.createNode("plugNodeA", nodeType=TestPlugNode)
	# print(PlugNode.inputPlugRoot(plugNode))
	# print(TestPlugNode.plugChildren(TestPlugNode.inputPlugRoot(plugNode)))


if __name__ == '__main__':
	makePlugNode()


