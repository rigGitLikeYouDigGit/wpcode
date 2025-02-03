from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm.tool.leyline import LLGraph, LLPoint, LLEdge

if __name__ == '__main__':

	g = LLGraph("testManifold")

	ptA = LLPoint("ptA", graph=g)
	ptB = LLPoint("ptB", graph=g)

	edge = LLEdge("edgeA", parents={ptA.name : {},
	                                ptB.name : {}},
	              graph=g)

	ptC = LLPoint("ptc", graph=g)

	ptC.addParent(edge)

	edge2 = LLEdge("edgeB", graph=g)
	edge2.addParents((ptA, edge))

	ptC.addParent(edge2)

	#print(edge)
	g.addElements([ptA, ptB, edge, ptC, edge2])

	log("graph path")
	nodeSeq = g.dirtyGraph.getDirtyNodesToEval(edge)
	#pprint.pp(list(nodeSeq))
	print([i.name for i in nodeSeq])
	nodeSeq = g.dirtyGraph.getDirtyNodesToEval(ptC)
	#pprint.pp(list(nodeSeq))
	print([i.name for i in nodeSeq])





