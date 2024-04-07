
from __future__ import annotations
import typing as T

import networkx as nx

from wplib import log

from wptree import Tree, TreeInterface

from chimaera.node import ChimaeraNode, NodeAttrRef, NodeAttrWrapper

"""experiments in chimaera graph execution - 
rich structure of the graph itself is already captured in
nodes - 
here we need to know what to execute if a given node has
to resolve.


individual tree branches can be nodes in the graph,
based on their inputs and connections - 

a tree is marked clean if it and all its branches are clean

"""

def inputNodeUids(node:ChimaeraNode)->set[str]:
	"""get all node uids that are inputs to this node
	"""
	uids = set()
	for key, wrapper in node._attrMap.items():
		for branch in wrapper.incomingTreeExpanded().allBranches():
			for attrRef in branch.value: #type:NodeAttrRef
				uids.add(attrRef.uid)
	return uids



def graphFromNodes(nodes:T.Iterable[ChimaeraNode])->nx.DiGraph:
	"""combined history of given nodes - no guarantee that
	they are connected in the graph

	consider setting this off with a threaded job whenever a node attribute changes
	"""
	graph = nx.DiGraph()

	visited = set()
	loopNodes = set()
	nodes = set(ChimaeraNode(node) for node in nodes)
	startNodes = set(nodes)

	graph.add_nodes_from(node.uid for node in nodes)

	while nodes:
		node = nodes.pop()
		if node.uid in visited:
			if node.uid in startNodes:
				pass
			else:
				loopNodes.add(node.uid)
			continue
		visited.add(node.uid)
		inputs = inputNodeUids(node)
		nodes.update(inputs)
		graph.add_nodes_from(uid for uid in inputs)
		graph.add_edges_from((uid, node.uid) for uid in inputs)
	return graph

