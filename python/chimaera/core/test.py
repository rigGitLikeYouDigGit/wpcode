
import pprint

from wplib.object import DirtyGraph, DirtyNode

from chimaera.core.node import ChimaeraNode
from chimaera.core.graph import ChimaeraGraph



# graph = ChimaeraGraph()
# node = graph.createNode("topNodeA")
# print(node)
#
# pprint.pprint(graph.data)
#
# pprint.pprint(node.dataBlock())
#
# childGraph = graph.createNode("childGraph", ChimaeraGraph)
#
# print(childGraph)
# print(childGraph.parent())
# pprint.pp(childGraph.dataBlock())
# pprint.pp(graph.data)


# execution
"""
consider a node with ref {"vals" : "n:a or n:b or n:c"}

"""

nodeA = DirtyNode("a")
nodeB = DirtyNode("b")
nodeC = DirtyNode("c")

dNodes = {nodeA, nodeB, nodeC}

nodeC.getDirtyNodeAntecedents = lambda: [nodeA, nodeB]
nodeB.getDirtyNodeAntecedents = lambda: [nodeA]



dGraph = DirtyGraph()

dGraph.addNodesAndPrecedents(dNodes)
print(nodeC.getDirtyNodeAntecedents())
print(nodeB.getDirtyNodeAntecedents())
pprint.pp(dGraph.nodes())

# for i in dGraph.nodes:
# 	print(i, i.dirtyState)

print(dGraph.edges)

print(dGraph.earliestDirtyNodes(dNodes))
nodeA.dirtyState = False
print(dGraph.earliestDirtyNodes(dNodes))



