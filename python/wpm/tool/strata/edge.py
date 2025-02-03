from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


from wplib.object import UidElement, DirtyNode, DirtyGraph
from wpm.tool.leyline.lib import DictModelled
from wpm.tool.leyline.element import LLElement

if T.TYPE_CHECKING:
	from .graph import LLGraph
	from .point import LLPoint

class LLEdge(LLElement):
	"""could debate if this should be called an edge
	or just a line

	nurbs curve between 2 llPoints living in scene,
	able to be reshaped by hand

	curve between llEdges saves parametres, offsets in framed-space
	curve across a face saves its parametres on that (somehow)

	parent data is dict of
	{ parent id : { rich data } }
	"""

	def __init__(self,
				name:str, # take name as unique id for all ll elements for now
	             parents:dict[str, dict]=None, # could be points, edges, faces etc
	             graph:LLGraph=None
	             ):
		LLElement.__init__(self,
		                   name, parents,
		                   graph)

