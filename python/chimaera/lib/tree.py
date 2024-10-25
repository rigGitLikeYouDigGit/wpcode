

from __future__ import annotations
import typing as T

from wplib.constant import MAP_TYPES, SEQ_TYPES, STR_TYPES, LITERAL_TYPES, IMMUTABLE_TYPES

from wptree import Tree, TreeInterface


"""copying overlay and composition functions here since
all of chimaera runs on them"""

def combineValues(baseBranch:TreeInterface,
                  overlayBranch:TreeInterface)->T.Any:
	"""test for catch-all to merge 2 disparate values
	- if both are sequences, extend? overlay? override?
	too complex for automatic, has to be directed

	eventually maybe refit this for full broadcasting, but
	this is good enough to test
	"""
	baseValue = baseBranch.value
	overlayValue = overlayBranch.value
	if baseValue is None:
		return overlayValue
	if overlayValue is None:
		return baseValue

	if isinstance(baseValue, SEQ_TYPES) and isinstance(overlayValue, SEQ_TYPES):
		return list(baseValue) + list(overlayValue)
	if isinstance(baseValue, SEQ_TYPES):
		return list(baseValue) + [overlayValue]
	if isinstance(overlayValue, SEQ_TYPES):
		return [baseValue] + list(overlayValue)
	if isinstance(baseValue, MAP_TYPES) and isinstance(overlayValue, MAP_TYPES):
		baseValue.update(overlayValue)
		return baseValue
	return overlayValue


def overlayTreeInPlace(baseTree:TreeInterface, overlay:TreeInterface,
				 resolveValueFn:T.Callable[[TreeInterface, TreeInterface], T.Any]=combineValues,
                       				 mode="union"
				 )->TreeInterface:
	"""apply overlay to baseTree in place,
	may change the structures of both trees
	if mode == "union", add all missing branches found in overlay to base tree
	if mode == "intersection", only update branches that exist in base tree

	"""
	# base tree nodes to visit
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

