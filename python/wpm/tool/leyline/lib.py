from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wplib.sequence import resolveSeqIndex
from wplib.serial import Serialisable

class DictModelled(dict, Serialisable):
	"""
	move this to wplib.object if it works well
	consider maybe defining attributes
	as type hints, checking against them on setattr, etc"""
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def __setattr__(self, key, value):
		if key in self.__dict__:
			self.__dict__[key] = value
		else:
			self[key] = value

	def __getattr__(self, item):
		if item in self.__dict__:
			return self.__dict__[item]
		return self[item]

	def encode(self, encodeParams:dict=None) ->dict:
		return dict(self)

	@classmethod
	def decode(cls, serialData:dict, decodeParams:dict=None) ->T.Any:
		return cls(**serialData)

def insertInDict(d:dict, index:int, k:T.Any, v:T.Any):
	"""reorders dict in-place to add the given value at the given index
	"""
	pairTuples = list(d.items())
	pairTuples.insert(resolveSeqIndex(index, len(pairTuples)), (k, v))
	d.clear()
	d.update(pairTuples)
	return d
