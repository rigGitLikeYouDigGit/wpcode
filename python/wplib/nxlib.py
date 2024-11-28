from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import networkx as nx

def multiGraphEdgesMatchingKey(graph:nx.MultiGraph,
                          node:T.Any,
                          key:str):
	assert node in graph
	return (i for i in graph.edges(node, keys=True)
	        if i[2] == key)

def multiGraphAdjacentNodesForKey(graph:nx.MultiGraph,
                                  node:T.Any,
                                  key:str):
	# log("multiGraphAdjNodes")
	# log(graph, node, key)
	assert node in graph
	return (i[1] for i in multiGraphEdgesMatchingKey(graph, node, key))

if __name__ == '__main__':

	class TestPoint:pass
	class TestEdge:pass
	g = nx.MultiGraph()
	ptA = TestPoint()
	ptB = TestPoint()
	g.add_node(ptA)
	g.add_node(ptB)
	e = TestEdge()
	g.add_edge(ptA, ptB, key="connectEnd")
	g.add_edge(ptA, e, key="connectLine")
	g.add_edge(ptB, e, key="connectLine")

	print(g.edges(ptA))
	print(g.edges(ptA, keys=True))
	print(*g.neighbors(ptA,))

	print(*multiGraphEdgesMatchingKey(g, ptA, "connectLine"))
	print(*multiGraphEdgesMatchingKey(g, ptA, "connectEnd"))
	print(*multiGraphAdjacentNodesForKey(g, ptA, "connectLine"))
	print(*multiGraphAdjacentNodesForKey(g, e, "connectLine"))