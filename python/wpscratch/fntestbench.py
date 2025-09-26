from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import numpy as np

from wplib.object import UserDecorator
"""
side test for a simple way to get pseudo-random time segments in a shader,
where all you have is a steadily increasing value of T

S(T) -> start value of T for this segment
L(T) -> segment length in T for this segment

S(T) + L(T) = S( S(T) + L(T) )
this must hold

"""


sFns = []
lFns = []

class register(UserDecorator):

	def wrapFunction(self,
	                 targetFunction:callable,
	                 decoratorArgsKwargs:(None, tuple[tuple, dict])=None) ->function:
		if not decoratorArgsKwargs:
			return targetFunction
		if "s" in decoratorArgsKwargs[0]:
			sFns.append(targetFunction)
		if "l" in decoratorArgsKwargs[0]:
			lFns.append(targetFunction)
		return targetFunction


@register("s")
def sBase(t:float, lFn):
	return np.floor(t)

@register("l")
def lBase(t:float):
	return 1.0

def fromiter(x, f, *args):
	return np.fromiter((f(xi, *args) for xi in x), x.dtype)
def checkCondition(sFn, lFn):
	vals = np.arange(1, 100, 0.31)
	sVals = fromiter(vals, sFn)
	lVals = fromiter(vals, lFn)

	summed = sVals + lVals

	compound = fromiter(summed, sFn)

	# print(vals)
	# print(sVals)
	# print(summed)
	# print(compound)
	result = np.max(np.abs(summed - compound))
	if(result < 0.001):
		print("FOUND VALID PAIR:", sFn, lFn)


if __name__ == '__main__':
	#checkCondition(sBase, lBase)

	for sFn in sFns:
		for lFn in lFns:
			checkCondition(sFn, lFn)




