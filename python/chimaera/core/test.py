
import pprint

from wplib.object import DirtyGraph, DirtyNode

from chimaera.core.node import ChimaeraNode
#from chimaera.core.graph import ChimaeraGraph
from chimaera.core.exegraph import ChimaeraDirtyGraph


graph = ChimaeraNode()
node = graph.createNode("topNodeA")
print(node)

pprint.pprint(graph.data)

pprint.pprint(node.dataBlock())

childGraph = graph.createNode("childGraph", ChimaeraNode)

print(childGraph)
print(childGraph.parent())
pprint.pp(childGraph.dataBlock())
pprint.pp(graph.data)


eGraph = ChimaeraDirtyGraph()
eGraph.addChimaeraNode(graph)

print(eGraph.nodes)


