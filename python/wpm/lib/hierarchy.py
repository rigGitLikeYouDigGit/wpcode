
from __future__ import annotations
import typing as T

from wpm import om


def iterDagChildren(root:om.MObject, includeRoot=False)->T.Iterator[om.MObject]:
	"""generator to iterate over dag children of root"""
	if includeRoot:
		yield root
	dag = om.MDagPath.getAPathTo(root)
	for i in range(dag.childCount()):
		yield dag.child(i)
		yield from iterDagChildren(dag.child(i), includeRoot=False)

def relativeDagTokens(fromMDagPath:om.MDagPath, toMDagPath:om.MDagPath)->list[str]:
	"""return relative dag tokens from one dag path to another -
	with ".." representing parent of previous token
	"""
	fromTokens = fromMDagPath.fullPathName().split("|")[1:]
	toTokens = toMDagPath.fullPathName().split("|")[1:]

	#print("fromTokens", fromTokens, "toTokens", toTokens)

	for i in range(min(len(fromTokens), len(toTokens)) + 1):
		fromSet = set(fromTokens[:i])
		toSet = set(toTokens[:i])

		#print("fromSet", fromSet, "toSet", toSet, fromSet == toSet)

		# the first time the tokens are not equal indicates divergence
		if fromSet != toSet:
			break

	parentChain = [".."] * (len(fromTokens) - i)
	childChain = toTokens[i:]
	#print("relative dag tokens", parentChain + childChain)
	return parentChain + childChain



