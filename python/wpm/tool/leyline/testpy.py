from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm.tool.leyline import LLGraph, LLPoint, LLEdge

if __name__ == '__main__':

	g = LLGraph("testManifold")

	ptA = LLPoint("ptA")
	ptB = LLPoint("ptB")

	edge = LLEdge("edgeA", parents={ptA.name : {},
	                                ptB.name : {}})
	#print(edge)
	g.addElements([ptA, ptB, edge])

	log("graph path")
	nodeSeq = g.dirtyGraph.getDirtyNodesToEval(edge)
	pprint.pp(list(nodeSeq))





