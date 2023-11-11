
from __future__ import annotations
import typing as T

from collections import defaultdict, namedtuple
from typing import TypedDict

from wplib.object import DeepVisitor
from wplib.serial import Serialisable
"""test for a simpler and more general approach - 

2 passes: 
	- parse with ParseData return functions
	
	- evaluate with EvalData return functions
"""

class ParseData(TypedDict):
	"""aux data for parsing structure expressions"""

class EvalData(TypedDict):
	"""aux data for evaluating structure expressions"""

class Expression(Serialisable):
	"""a structure expression -
	check if structure is fully static with parsing pass
	"""

	def __init__(self, structure):
		self.structure = structure
		self.parsedStructure = None
		self.evaluatedStructure = None

	@classmethod
	def _encodeObject(cls, obj, encodeParams:dict):
		return {"structure" : obj.structure}

	@classmethod
	def _decodeObject(cls, serialCls:type[Expression], serialData:dict,
	                 decodeParams:dict, formatVersion=-1) ->Expression:
		return serialCls(serialData["structure"])


class VisitParseExpOp(DeepVisitor.DeepVisitOp):
	pass

class VisitEvalExpOp(DeepVisitor.DeepVisitOp):
	pass





