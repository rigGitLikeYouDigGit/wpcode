
from __future__ import annotations
import typing as T
import numpy as np
from scipy.sparse import dok_array, csc_array
from wpm import cmds, om, WN

"""skins are super powerful to work with in code, but 
making it robust to a big scene is tough

check for empty matrix indices, check empty weights,
static bind matrices vs live, etc
"""


def getSkinArraySparse(skFn:om.MFnDependencyNode)->csc_array:
	"""return skin weights as a sparse array, giving massive memory savings
	"""
	matrixPlug = skFn.findPlug("matrix", False)
	influenceIndices = matrixPlug.getExistingArrayAttributeIndices()
	maxInfluenceIndex = max(influenceIndices)

	# set up plugs used in iteration
	weightListPlug = skFn.findPlug("weightList", False)
	weightListMObj = weightListPlug.attribute()

	nVertices = max(weightListPlug.getExistingArrayAttributeIndices()) + 1

	weightsPlug = skFn.findPlug("weights", False)
	leafPlug = om.MPlug(weightsPlug)

	# array for final weights
	# dok used for fast construction
	weightArray = dok_array((nVertices, maxInfluenceIndex + 1), dtype=float)

	# for vtx in range(nVertices):
	for vtx in weightListPlug.getExistingArrayAttributeIndices():
		weightsPlug.selectAncestorLogicalIndex(vtx, weightListMObj)
		leafPlug.selectAncestorLogicalIndex(vtx, weightListMObj)

		for index in weightsPlug.getExistingArrayAttributeIndices():
			leafPlug.selectAncestorLogicalIndex(index)
			weightArray[vtx, index] = leafPlug.asFloat()

	# convert to final sparse type
	arr = weightArray.tocsc()
	arr.sort_indices()
	return arr


def setSkinArraySparse(skFn:om.MFnDependencyNode,
				   weightArr:csc_array):
	"""key to scipy sparse structures are 2 arrays describing shape:
	arr.indices and arr.indptr

	arr = dok_array((3, 3))
		>> [ [0, 0, 0],
			 [0, 0, 0],
			 [0, 0, 0], ]
	csc = arr.tocsc()
	csc.sort_indices()
	>> [0 0 0 0] ptr
	>> [] indices

	arr[0, 0] = 1.0
	>> [0 1 1 1] ptr
	>> [0] indices

	arr[2, 2] = 1.0
	>> [0 1 1 2] ptr
	>> [0 2] indices

	arr[2, 1] = 1.0
	>> [0 1 2 3] ptr
	>> [0 2 2] indices

	and so on -
	ptr is always 1 more than nRows, and

	indices[ ptr[row] : ptr[row + 1] ]
	gives all existing indices in that row
	"""
	nVertices, nWeights = weightArr.shape

	weightListPlug = skFn.findPlug("weightList", False)
	weightListMObj = weightListPlug.attribute()

	weightsPlug = skFn.findPlug("weights", False)
	leafPlug = om.MPlug(weightsPlug)

	# for vtx in range(nVertices):
	for vtx in range(nVertices):
		# check if row exists in sparse array - if not, skip
		if (weightArr.indptr[vtx] == weightArr.indptr[vtx + 1]): # no entries in this row
			continue
		weightsPlug.selectAncestorLogicalIndex(vtx, weightListMObj)
		leafPlug.selectAncestorLogicalIndex(vtx, weightListMObj)

		# first set all existing weight plugs to zero
		for index in weightsPlug.getExistingArrayAttributeIndices():
			leafPlug.selectAncestorLogicalIndex(index)
			leafPlug.setFloat(0.0)
		weightsPlug.setNumElements(weightArr.indptr[vtx] - weightArr.indptr[vtx + 1])

		for index in weightArr.indices[ weightArr.indptr[vtx] : weightArr.indptr[vtx + 1]]:
			leafPlug.selectAncestorLogicalIndex(index)
			leafPlug.setFloat(weightArr[vtx, index])


class SkinIndexMap:
	"""CHANGE THE NAME to something more expressive -
	helper mainly to manage skin indices and remapping
	between local skin groups and the full scene

	TODO:
	 initialising from weight arrays
	 maybe abstract object to work with per-element weights
	 is it worth abstracting this object to manage indices
	 maybe just put this straight on to the node objects

	"""

	def __init__(self, node:WN.SkinCluster
				 ):
		self.node : WN.SkinCluster = WN.SkinCluster(node)

	def activeIndices(self)->np.array:
		"""return all active influence indices -
		"""
		return np.array(self.node.bindPreMatrix_.indices())



	def globalSkinIndexMap(self, skins=()):
		"""return a map of all associated influences
		to a single unique index
		if no skins specified, use all skins in the scene
		"""

class SkinIndexGroup:
	"""map indices between different skincluster nodes"""

class Skin:
	"""determine main skin influences from bindPreMatrix neutral plugs -
	the same transform might drive 2 influences, or an influence could have
	no sriving transform at all"""

def getSkinCluster(node:EdNode)->EdNode:
	"""get skincluster connected to node,
	or node if it is a skincluster already"""
	node = EdNode(node)
	if isinstance( node.MFn, oma.MFnSkinCluster ):
		return node
	# return latest skincluster in history
	skins = node.shape.history(EdNode.Types.kSkinCluster)
	if not skins:
		return None
	return skins[-1]

# region influence indices, mapping, unification
# welcome
# to
# hell
def globalOrderPlugTrees(plugTrees:list[PlugTree])->tuple[PlugTree]:
	"""return consistent ordering for given plug trees"""
	return tuple(sorted(tuple(plugTrees)))


def skinIndexInfluencePlugMap(skc:EdNode)->dict[int, PlugTree]:
	""" for local-scale functions we need to know the exact effects
	on the final skin - that means following all logical array indices properly

	return map of
	{ local influence logical index : influence plug }

	"""
	infIndexPlugMap = {}
	skc = EdNode(skc)
	matTree = skc("matrix")
	drivers = matTree.drivers()
	for driver, driven in drivers.items():
		infIndexPlugMap[driven.name] = driver
	return infIndexPlugMap




# endregion


# region skin weights
# this is the easy bit
def getSkinArray(skc:EdNode)->np.ndarray:
	"""retrieve the weights of the given skincluster as 2d numpy array
	no processing done on indices.
	Logical influence indices matter for gathering weights -
	you should ensure that all arrays match physically, prior to this, but
	here we work by whatever logical indices exist on node

	optimisation here is to use persistent plugs for each level of weight tree,
	and set internal indices rather than generating new plug objects

	return dense array of [ vertex index : [ weight index : weight value ] ]
	"""
	skc = EdNode(skc)
	matrixPlug = skc("matrix").MPlug
	influenceIndices = matrixPlug.getExistingArrayAttributeIndices()
	maxInfluenceIndex = max(influenceIndices)
	nIndices = len(influenceIndices)

	# set up plugs used in iteration
	weightListTree = skc("weightList")
	weightListPlug = weightListTree.MPlug
	weightListMObj = weightListPlug.attribute()

	nVertices = max(weightListPlug.getExistingArrayAttributeIndices()) + 1

	weightsTree = skc("weights")
	weightsPlug = weightsTree.MPlug
	leafPlug = om.MPlug(weightsPlug)

	# array for final weights
	weightArray = np.zeros((nVertices, maxInfluenceIndex + 1), dtype=float)

	#for vtx in range(nVertices):
	for vtx in weightListPlug.getExistingArrayAttributeIndices():
		# elPlug = weightListPlug.elementByPhysicalIndex(vtx)
		# weightsPlug = elPlug.child(0)
		weightsPlug.selectAncestorLogicalIndex(vtx, weightListMObj)
		leafPlug.selectAncestorLogicalIndex(vtx, weightListMObj)

		for index in weightsPlug.getExistingArrayAttributeIndices():
			leafPlug.selectAncestorLogicalIndex(index)
			weightArray[vtx, index] = leafPlug.asFloat()

	return weightArray

# endregion

def getSkinInfluences(skc:EdNode)->tuple[PlugTree]:
	return tuple(skinIndexInfluencePlugMap(skc).values())

def resetSkinCluster(skc:EdNode, influenceList:list[PlugTree]):
	"""coerce skin to have only the given influences,
	connected literally as physical indices"""

	# first disconnect and reset everything
	skc("matrix").breakConnections()
	# reset neutral matrices
	#plug:PlugTree
	for plug in skc("bindPreMatrix").branches:
		plug.set(om.MMatrix())

	for i, plug in enumerate(influenceList):
		plug.use(plug, skc("matrix", i))


def setWeightArray(skc, weightArr: np.ndarray = None):
	"""assumed to be fully dense"""
	nVertices, nWeights = weightArr.shape

	weightListPlug = findPlug("weightList", False)
	weightListMObj = weightListPlug.attribute()

	# nVertices = max(weightListPlug.getExistingArrayAttributeIndices()) + 1

	weightsPlug = skFn.findPlug("weights", False)
	leafPlug = om.MPlug(weightsPlug)

	# for vtx in range(nVertices):
	for vtx in range(nVertices):
		weightsPlug.selectAncestorLogicalIndex(vtx, weightListMObj)
		leafPlug.selectAncestorLogicalIndex(vtx, weightListMObj)

		# for index in weightsPlug.getExistingArrayAttributeIndices():
		for index in range(nWeights):
			leafPlug.selectAncestorLogicalIndex(index)
			leafPlug.setFloat(weightArr[vtx, index])
	return weightArr

def poolSkinClusters(skcList:list[EdNode]):
	"""top-level fire-and-forget function to unify all influences across all given
	skinclusters, ready for array operations.
	This sets each influence to its unique global index, across all skins

	We also preserve any existing skin weights,
	and we assume that the current positions of all joints correspond to the neutral bind matrices
	"""
	skcIndexInfMaps = []
	skcLocalWeightArrays = []

	# gather all influences across all skins
	allInfluences = set()
	for skc in skcList:
		skcIndexInfMap = skinIndexInfluencePlugMap(skc)
		skcIndexInfMaps.append(skcIndexInfMap)
		allInfluences.update(skcIndexInfMap.values())

		# gather local weights
		skcLocalWeightArrays.append(getSkinArray(skc))
	orderedInfluences = globalOrderPlugTrees(allInfluences)

	# get local-to-global index map for each skin
	# this is even more complicated since we still need to respect
	# semantic indices for local maps
	localToGlobalIndexMaps = []

	for skcIndexInfMap in skcIndexInfMaps:
		#localSemanticToGlobalArray =
		localToGlobalIndexMap = np.zeros(max(skcIndexInfMap) + 1, dtype=int)
		for i, (localIndex, influence) in enumerate(skcIndexInfMap.items()):
			localToGlobalIndexMap[localIndex] = orderedInfluences.index(influence)
		localToGlobalIndexMaps.append(localToGlobalIndexMap)

	# # connect all skin influences
	# for i, skc in enumerate(skcList):
	#
	#
	# for i, influence



	#globalweights[localmap] += localweights # adding lets us take unused influences in stride


