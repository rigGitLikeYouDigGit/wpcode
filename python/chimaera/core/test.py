
import pprint

from chimaera.core.node import ChimaeraNode
from chimaera.core.graph import ChimaeraGraph


graph = ChimaeraGraph()
node = graph.createNode("topNodeA")
print(node)

pprint.pprint(graph.data)

pprint.pprint(node.dataBlock())

childGraph = graph.createNode("childGraph", ChimaeraGraph)

print(childGraph)
print(childGraph.parent())
pprint.pp(childGraph.dataBlock())
pprint.pp(graph.data)


