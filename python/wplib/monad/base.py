from __future__ import annotations
import types, typing as T
import pprint
import typing

from wplib import log
from typing import TypedDict

from wplib import sequence
from wplib.sentinel import Sentinel

"""simple system to set up function pipelines as monads, mainly 
for the purpose of UI systems.
Some overlap here with object.multiobject in terms of invoking methods
on groups of valid objects
"""

EVAL = lambda x : x() if isinstance(x,
    (types.FunctionType, types.MethodType)) else x

def EVALR(x):
	""" need some way to mark branches as eval'd once, don't need to keep
	iterating after first pass

	(could have crazy system where we keep one master copy as the generator,
	and keep on re-evaling all its branches at each step,
	"""
	if isinstance(x, dict):
		return {EVALR(k) : EVALR(v) for k, v in x.items()}
	elif isinstance(x, list):
		return [EVALR(i) for i in x]
	return EVAL(x)

class MonadDriveData(TypedDict):
	monad : Monad
	target : typing.Callable

class Monad:
	"""i wonder if other languages have specific names for these
	like rust, c, go

	unsure if methods on this object should be module-level functions instead,
	would remove danger of collision entirely

	this might also overlap with exp system
	"""
	def __init__(self, *args, **kwargs) -> None:
		"""v is any seed object for the monad, either static value or
		callable"""
		self.args = args
		self.kwargs = kwargs

	def __getattr__(self, item):
		"""getattr is the main way to build up the monad, it returns a new
		monad with the new value being the result of calling the previous
		value with the new attribute as a method"""
		return Monad(lambda : getattr(EVALR(self.args[0]), item))

	def __call__(self, *args, **kwargs):
		return Monad(lambda : self.args[0](*args, **kwargs))

	def drive_(self, fn:typing.Callable):
		"""create a record connecting this end monad to the target callable -
		this doesn't automatically fire target when this monad evals.
		Maybe we store this on the monad too, but wait for use case til
		we add state here"""
		return MonadDriveData(
			monad=self, target=fn
		)


def chainIsReversible_(self):
	pass


class Each(Monad):
	"""iterate over only flat values for now, items for lists,
	(k v) tuples for dicts
	"""

"""for ui, we need a way to explicitly define reversible
operations. lenses is a cool library but I'm gonna say it's a bit complex for
now
"""

#
# presOp = Iso(forwards=modifyFn, backwards=restoreFn)
#
#
# reverseMap = {
# 	str.upper: str.lower,
# }
