from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wplib.object import UidElement, DirtyNode, DirtyGraph
from wpm.tool.leyline.lib import DictModelled
from wpm.tool.leyline.element import LLElement

if T.TYPE_CHECKING:
	from .graph import LLGraph

class LLPoint(LLElement):
	"""unsure if it's worth a single base class for LL elements -
	all using the same interface for parents, even though points here should only
	have one parent"""

	def __init__(self,
				name:str, # take name as unique id for all ll elements for now
				 parents: dict[str, dict] = None,  # could be points, edges, faces etc
				 graph: LLGraph = None,
	             pos=(0, 0, 0),
	             uiPos=(0, 0),
	             ):
		LLElement.__init__(self,
		                   name,
		                   parents,
		                   graph,
		                   pos=pos,
		                   uiPos=uiPos)
		log("pt", self.name, "parents", self.parents)


if __name__ == '__main__':
	from wplib.serial import serialise, deserialise

	ptA = LLPoint("ptA")
	assert type(ptA) == LLPoint

	data = serialise(ptA)
	log("data", data)
	log(type(data))
	assert type(data) == dict

	newPt = deserialise(data)
	assert type(data) == LLPoint



