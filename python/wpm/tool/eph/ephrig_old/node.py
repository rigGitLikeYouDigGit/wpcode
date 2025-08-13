
"""
abstract system for the graph structure -

for now, use simply directed structure from skeleton

a single master graph is maintained as source of truth for topology,
and for each user input selected, a child graph is created,
covering only the affected nodes
User transformations are summed across each graph and combined
to give final effects on maya nodes

Simply:

			/ --- ToolGraph \
MasterGraph  -----------> combine ---> result
			\ --- ToolGraph /

for finding the path of the

"""

from collections import defaultdict, namedtuple
import site
site.addsitedir("C:\Python37\Lib\site-packages")
import networkx as nx # gonna be awkward if nxt gets involved

from networkx import DiGraph

from edRig.palette import *


#EphEdge = namedtuple("EphEdge", ["src", "dest", "data"])

class EphNode(object):
	"""
	Module encapsulating one part of the eph system

	SHAMELESS copy from Mme Anzovin
	we assume the node object will be added to graph externally"""
	def __init__(self, name
	             ):
		self.name = name

		self.parent = None
		self.children = []
		self.graph = None #type:DiGraph

	def __hash__(self):
		return id(self)

	@property
	def neighbours(self):
		return [self.parent] + self.children

	def inEdges(self):
		return self.graph.in_edges(self)

	def outEdges(self):
		return self.graph.out_edges(self)

	def serialise(self):
		return {}







