
from __future__ import annotations
import typing as T

from wplib import log

from .interface import TreeInterface

"""library for operations on trees"""


def overlayTreeInPlace(baseTree:TreeInterface, overlay:TreeInterface,
				 resolveValueFn=lambda baseBranch, overlayBranch: overlayBranch.value,
                       				 mode="union"
				 )->TreeInterface:
	"""apply overlay to baseTree in place,
	may change the structures of both trees
	if mode == "union", add all missing branches found in overlay to base tree
	if mode == "intersection", only update branches that exist in base tree

	"""
	# base tree nodes to visit
	# toVisitOverlay = list(overlay.allBranches(
	# 		includeSelf=False,
	# 		depthFirst=True,
	# 		topDown=True))
	toVisitOverlay = [overlay]
	while toVisitOverlay:

		overlayBranch = toVisitOverlay.pop(0)
		overlayPath = overlayBranch.relAddress(fromBranch=overlay)
		baseBranch = baseTree
		branchMissing = False
		i = 0
		for i, token in enumerate(overlayPath):
			baseTest = baseBranch.getBranch(token)
			if baseTest is None: # missing branch
				branchMissing = True
				break
			baseBranch = baseTest

		# log("overlayTreeInPlace", "overlayBranch", overlayBranch)
		# log("overlayTreeInPlace", "baseBranch", baseBranch)
		# log("overlayTreeInPlace", "branchMissing", branchMissing)
		overlayBranch = overlay(overlayPath[:i+1])
		if branchMissing:
			#remove added overlays from toVisitOverlay
			for i in overlayBranch.allBranches(includeSelf=True):
				#toVisitOverlay.remove(i)
				if i in toVisitOverlay:
					toVisitOverlay.remove(i)

			if mode == "intersection":
				continue
			if mode == "union":
				# add branch to base tree
				# get parent branch and
				baseBranch.addBranch(overlayBranch)
				continue

		lookupBranch = baseBranch
		if lookupBranch is None:
			if mode == "union":
				baseTree.addBranch(overlayBranch)
				continue
				#lookupBranch = baseTree.getBranch(overlayBranch.relAddress(fromBranch=overlay))
			else:
				continue
		newValue = resolveValueFn(lookupBranch, overlayBranch)
		lookupBranch.setValue(newValue)

		# add children to toVisitOverlay
		toVisitOverlay.extend(overlayBranch.branches)
	return baseTree



def overlayTrees(overlays:list[TreeInterface],
                 resolveValueFn=lambda branchA, branchB: branchB.value,
                 mode="union",
                 )->TreeInterface:
	"""return new tree with overlay applied to baseTree"""
	baseTree = overlays[0]
	resultTree = baseTree.copy()
	for overlay in overlays[1:]:
		overlayTreeInPlace(resultTree, overlay.copy(), resolveValueFn=resolveValueFn,
		                   mode=mode)
	return resultTree




