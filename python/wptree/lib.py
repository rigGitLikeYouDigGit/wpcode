
from __future__ import annotations
import typing as T

from .interface import TreeInterface

"""library for operations on trees"""


def overlayTreeInPlace(baseTree:TreeInterface, overlay:TreeInterface,
				 resolveValueFn=lambda baseBranch, overlayBranch: overlayBranch.value,
                       				 addMissingBranches=True,
				 )->TreeInterface:
	"""apply overlay to baseTree in place"""
	for overlayBranch in overlay.allBranches(includeSelf=False,
	                                  depthFirst=True,
		topDown=True):

		lookupBranch = baseTree.getBranch(overlayBranch.relAddress(fromBranch=overlay))
		if lookupBranch is None:
			if addMissingBranches:
				baseTree.addChild(overlayBranch)
			else:
				continue
		newValue = resolveValueFn(lookupBranch, overlayBranch)
		lookupBranch.setValue(newValue)
	return baseTree



def overlayTrees(baseTree:TreeInterface, *overlays:TreeInterface,
                 resolveValueFn=lambda branchA, branchB: branchB.value,
                 addMissingBranches=True,
                 )->TreeInterface:
	"""return new tree with overlay applied to baseTree"""
	resultTree = baseTree.copy()
	for overlay in overlays:
		overlayTreeInPlace(resultTree, overlay, resolveValueFn=resolveValueFn,
		                   addMissingBranches=addMissingBranches)
	return resultTree




