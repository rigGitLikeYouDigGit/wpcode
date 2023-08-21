
import pprint

from wplib.object import DirtyGraph, DirtyNode

from chimaera.core.node import ChimaeraNode
#from chimaera.core.graph import ChimaeraGraph
from chimaera.core.exegraph import ChimaeraDirtyGraph


graph = ChimaeraNode()
node = graph.createNode("topNodeA")
print(node)

pprint.pprint(graph.dataBlock())

#pprint.pprint(node.dataBlock())

childGraph = graph.createNode("childGraph")

from chimaera.core.plugnode import PlugNode
pNode = childGraph.createNode("plugNode", PlugNode)
print(pNode)


eGraph = ChimaeraDirtyGraph.create()
eGraph.addChimaeraNode(graph)

print(eGraph.nodes)


