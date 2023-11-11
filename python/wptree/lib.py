
from __future__ import annotations
import typing as T

from .interface import TreeInterface

"""library for operations on trees"""


def overlayTreeInPlace(baseTree:TreeInterface, overlay:TreeInterface,
				 resolveValueFn=lambda baseBranch, overlayBranch: overlayBranch.value,
                       				 mode="union"
				 )->TreeInterface:
	"""apply overlay to baseTree in place
	if mode == "union", add all missing branches found in overlay to base tree
	if mode == "intersection", only update branches that exist in base tree

	"""
	for overlayBranch in overlay.allBranches(includeSelf=False,
	                                  depthFirst=True,
		topDown=True):

		lookupBranch = baseTree.getBranch(overlayBranch.relAddress(fromBranch=overlay))
		if lookupBranch is None:
			if mode == "union":
				baseTree.addChild(overlayBranch)
			else:
				continue
		newValue = resolveValueFn(lookupBranch, overlayBranch)
		lookupBranch.setValue(newValue)
	return baseTree



def overlayTrees(overlays:list[TreeInterface],
                 resolveValueFn=lambda branchA, branchB: branchB.value,
                 mode="union",
                 )->TreeInterface:
	"""return new tree with overlay applied to baseTree"""
	baseTree = overlays[0]
	resultTree = baseTree.copy()
	for overlay in overlays[1:]:
		overlayTreeInPlace(resultTree, overlay, resolveValueFn=resolveValueFn,
		                   mode=mode)
	return resultTree




