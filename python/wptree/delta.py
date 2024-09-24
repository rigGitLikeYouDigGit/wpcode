
""" delta extraction for primitive objects and tree
indices and consistent ordering in sequences was
a nightmare for a long time, but I think I have a
good solution to it
each atomic delta stores a "sticky" direction (0 or 1)
when a conflicting index is found in master,
stick index dictates which direction inserted delta moves
by default, prefer sticking deltas together in large blocks


not tackling anything other than trees yet

looking back on it now, I think I lost my mind for a second here

"""
from __future__ import annotations

from dataclasses import dataclass

import typing as T

from wplib.object.namespace import TypeNamespace
from wplib.delta import DeltaAtom, DeltaAid
from wptree.reference import TreeReference
#from wplib.object.event import TreeEvent

from wptree.main import Tree
from wptree.interface import TreeInterface


"""DELTAS
delta processing works relative to the current delta "root",
the topmost branch of the current tree which is tracking deltas
(or could simply be the root of the whole tree)

if a change is made within this delta tracking scope,
we serialise it as a delta atom

if a change originates from outside (eg a new branch is created
and added to the tracking tree ), we consider that a creation event, as far as the tree is concerned.

in the same way, if a branch is parented to a branch outside this delta's scope,
that branch is considered deleted

the logic and behaviour to actually apply a delta should really be handled
by the tree / object itself
which is annoying

primitive types obviously can't be extended this way - but is it valid to defer
to those objects here?



"""



@dataclass
class TreeDeltaAtom(DeltaAtom):
	branchRef : TreeReference

	# to support using actual branches instead of references
	def resolveRef(self, ref:(TreeReference, TreeInterface),
	               root)->TreeInterface:
		return ref.resolve(root) if isinstance(ref, TreeReference) else ref

	def references(self):
		"""return all TreeReferences used by this object"""
		return [i for i in self.__dict__.values() if isinstance(i, TreeReference)]

	# def do(self, targetRoot:TreeInterface):
	# 	"""consider adding mode to define system of reference to use, but
	# 	that should be set on each reference itself"""
	# 	# if no targetRoot is given, use the main branch
	# 	targetRoot = targetRoot or self.branchRef
	# 	super(TreeDeltaAtom, self).do(targetRoot)

@dataclass
class TreeNameDelta(TreeDeltaAtom):
	"""represents change to branch name
	since we're obviously changing names of branches,
	we need to modify the last token of the tree reference
	address when doing and undoing
	"""
	oldValue: str
	newValue : str

	def do(self, target:TreeInterface):
		""""""
		self.resolveRef(self.branchRef, target).setName(self.newValue)

	def undo(self, target:TreeInterface):
		"""to reverse name delta, replace last address token
		with delta result"""
		#print("undo delta")
		tempRef = self.branchRef.__copy__()
		# replace ref address last token with new name
		if tempRef.address:
			tempRef.address = tempRef.address[:-1] + [self.newValue]
		lookupBranch = self.resolveRef(self.branchRef, target)
		lookupBranch.setName(self.oldValue)


@dataclass
class TreeValueDelta(TreeDeltaAtom):
	"""change to branch value
	for now store the value, but later investigate doing a deeper
	diff on the result
	"""
	oldValue: T.Any
	newValue : T.Any

	def do(self, target:TreeInterface):
		""""""
		lookupBranch = self.resolveRef(self.branchRef, target)
		lookupBranch.setValue(self.newValue)

	def undo(self, target:TreeInterface):
		""""""
		lookupBranch = self.resolveRef(self.branchRef, target)
		lookupBranch.setValue(self.oldValue)

# intermediate base to avoid logic duplication
@dataclass
class _TreeCreationDeletionDeltaBase(TreeDeltaAtom):
	parentRef: TreeReference
	serialData: dict
	treeType: type = None
	preserveUid: bool = True

	def createTreeFromData(self, target: TreeInterface) -> TreeInterface:
		"""create a new tree, of either a type serialised in
		this delta, or the target root's defaultCls"""
		treeType: TreeInterface = self.treeType if self.treeType else target.defaultBranchCls()
		newTree = treeType.deserialiseSingle(self.serialData, preserveUid=self.preserveUid)  #
		if self.parentRef is not None:
			self.resolveRef(self.parentRef, target).addChild(newTree,  # index=self.index
			                                                     )
		return newTree

	def unlinkTreeByData(self, target: TreeInterface):
		"""if the given tree now has a parent, remove it
		otherwise there isn't much to do to 'delete' the tree in the
		strong sense"""
		branch = self.resolveRef(self.branchRef, target)
		branch.remove()


@dataclass
class TreeCreationDelta(_TreeCreationDeletionDeltaBase):
	"""holds logic for tree creation and deletion
	SERIALISE UID - use for unique delta objects, just like unique trees
	we sadly NEED to serialise parent and index, because on undoing a deletion,
	the branch must be reattached to the correct parent within the same
	operation
	"""

	def do(self, target:TreeInterface, refMode=TreeReference.Mode.Uid):
		"""create new tree and add as child"""
		newTree = self.createTreeFromData(target)
		return newTree

	def undo(self, target:TreeInterface=None):
		self.unlinkTreeByData(target)


@dataclass
class TreeDeletionDelta(_TreeCreationDeletionDeltaBase):
	"""creation but backwards"""
	def do(self, target:TreeInterface=None, refMode=TreeReference.Mode.Uid):
		self.unlinkTreeByData(target)
		pass
	def undo(self, target:TreeInterface=None):
		newTree = self.createTreeFromData(target)


@dataclass
class TreeMoveDelta(TreeDeltaAtom):
	"""track changes to a branch structure"""
	oldParentRef: TreeReference
	parentRef: (TreeReference, None)
	oldIndex: int
	newIndex:int

	def do(self, target: TreeInterface):
		branch = self.resolveRef(self.branchRef, target)
		parentBranch = self.resolveRef(self.parentRef, target)
		branch.remove()
		parentBranch.addChild(branch, self.newIndex)

	def undo(self, target: TreeInterface):
		branch = self.resolveRef(self.branchRef, target)
		parentBranch = self.resolveRef(self.oldParentRef, target)
		branch.remove()
		parentBranch.addChild(branch, self.oldIndex)

@dataclass
class TreePropertyDelta(TreeDeltaAtom):
	"""save entire dict for simplicity"""
	# key : str
	oldValue : dict
	newValue : dict

	def do(self, targetRoot:TreeInterface, refMode=TreeReference.Mode.Uid):
		# self.resolveRef(self.branchRef, targetRoot).setAuxProperty(self.key, self.newValue)
		self.resolveRef(self.branchRef, targetRoot)._properties = self.newValue

	def undo(self, targetRoot:TreeInterface, refMode=TreeReference.Mode.Uid):
		#self.resolveRef(self.branchRef, targetRoot).setAuxProperty(self.key, self.oldValue)
		self.resolveRef(self.branchRef, targetRoot)._properties = self.oldValue

treeDeltaClasses = (TreeNameDelta, TreeValueDelta, TreePropertyDelta, TreeMoveDelta, TreeCreationDelta, TreeDeletionDelta)

# collecting TypeNamespace - prefer importing this over every separate delta type
class TreeDeltas(TypeNamespace):
	Base = TreeDeltaAtom
	Value = TreeValueDelta
	Name = TreeNameDelta
	Property = TreePropertyDelta
	Move = TreeMoveDelta
	Create = TreeCreationDelta
	Delete = TreeDeletionDelta


# @dataclass
# class TreeDeltaEvent(TreeEvent):
# 	"""event passed when a delta is generated -
# 	by default we only consider instant deltas
# 	'data' attribute here will be list of delta atoms
# 	"""



def getCompareTargetTiesByUid(a:TreeInterface, b:TreeInterface):
	"""return individual tuples of tree branches to compare against?
	either side can also be creation or deletion delta if no matching branch is found
	for now everything is uids, but other modes could be used in future"""
	aUidMap = {i.uid : i for i in a.allBranches(includeSelf=True)}
	bUidMap = {i.uid : i for i in b.allBranches(includeSelf=True)}

	compareTies = []
	for aUid, aBranch in aUidMap.items():
		if not aUid in bUidMap:
			compareTies.append( (aBranch, TreeDeletionDelta(
				TreeReference(aBranch, mode=TreeReference.Mode.RelPath),
				TreeReference(aBranch.parent, mode=TreeReference.Mode.RelPath),
				aBranch.serialiseSingle(),
				type(aBranch)
			)))
		else:
			compareTies.append( (aBranch, bUidMap.pop(aUid)))

	# any remaining branches in bUidMap are newly created
	for bUid, bBranch in bUidMap.items():
		compareTies.append( (TreeCreationDelta(
			TreeReference(bBranch, mode=TreeReference.Mode.RelPath),
			TreeReference(bBranch.parent, mode=TreeReference.Mode.RelPath),
			bBranch.serialiseSingle(),
			type(bBranch)
		), bBranch))
	return compareTies

def compareBranches(a:TreeInterface, b:TreeInterface,
                    mode=None)->list[TreeDeltaAtom]:
	"""compare a single level of two trees - return any deltas between them
	only consider uids to determine creation / deletion -
	a higher-level process can combine these into move deltas if needed

	FOR NOW use uids, but consider other modes being uid agnostic, working on index,
	name
	"""
	deltas = []
	if a.name != b.name:
		deltas.append(TreeNameDelta(
			TreeReference(a, mode=TreeReference.Mode.RelPath), a.name, b.name	))

	#todo: find a way to recursively compare value objects with deepdiff
	if a._getRawValue() != b._getRawValue():
		deltas.append(TreeValueDelta(
			TreeReference(a, mode=TreeReference.Mode.RelPath), a.value, b.value))

	if a.auxProperties != b.auxProperties:
		deltas.append(TreePropertyDelta(
			TreeReference(a, mode=TreeReference.Mode.RelPath), a.auxProperties, b.auxProperties))

	# check for move
	if a.parent and b.parent:
		#print("a", a, a.index(), "b", b, b.index())
		if (a.parent.uid != b.parent.uid) or (a.index() != b.index()):
			deltas.append(TreeMoveDelta(
				branchRef=TreeReference(a, mode=TreeReference.Mode.RelPath),
				oldParentRef=TreeReference(a.parent, mode=TreeReference.Mode.RelPath),
				parentRef=TreeReference(b.parent, mode=TreeReference.Mode.RelPath),
				oldIndex=a.index(),
				newIndex=b.index()
			))

	return deltas

def compareTrees(a:TreeInterface, b:TreeInterface):
	"""top-level, handles gathering and combining deltas,
	returning a final list of deltaAtoms"""
	#print("compare trees")
	#print([i.name for i in a.allBranches()])
	#print([i.name for i in b.allBranches()])
	leafDeltas = []
	compareTargetTies = getCompareTargetTiesByUid(a, b)
	for src, dest in compareTargetTies:
		if isinstance(src, DeltaAtom):
			leafDeltas.append(src)
			continue
		elif isinstance(dest, DeltaAtom):
			leafDeltas.append(dest)
			continue
		leafDeltas.extend(compareBranches(src, dest))
	return leafDeltas


def branchToSyncForDelta(delta:TreeDeltaAtom, relativeRoot:Tree)->Tree:
	"""some deltas detail a change on the main branch,
	some directly below it
	return the branch that must be synced to display the
	given delta

	if multiple branches deleted at once, the parents of leaves might
	have been removed as well - in these cases, return None
	"""
	if isinstance(delta, TreeNameDelta):
		"""this delta defines a change in name, so its reference naturally points to
		the previous name of the tree"""
		if not delta.branchRef.address: # tree root
			return delta.branchRef.resolve(relativeRoot)
		newRef : TreeReference = delta.branchRef.__copy__()
		newRef.address[-1] = delta.newValue
		return newRef.resolve(relativeRoot)

	if isinstance(delta, (TreeValueDelta, TreePropertyDelta)):


		return delta.branchRef.resolve(relativeRoot)
	if isinstance(delta, (TreeCreationDelta, TreeDeletionDelta)):

		try:
			return delta.parentRef.resolve(relativeRoot)
		except:
			return None

	if isinstance(delta, (TreeMoveDelta, )):
		return delta.parentRef.resolve(relativeRoot)
	raise TypeError("unknown delta type", delta, type(delta))


class TreeDeltaAid(DeltaAid):
	forTypes = (TreeInterface, )

	@classmethod
	def gatherDeltas(cls, baseObj:TreeInterface, newObj:TreeInterface
	                 ) ->list[DeltaAtom]:
		return compareBranches(baseObj, newObj)




# class TreeDeltaContext(DeltaContext):
# 	"""delta context specialised for trees"""
#
# 	# scratch tree type should be as simple as possible
# 	deltaTreeType = None # specify, or will be set to default Tree on init
# 	@classmethod
# 	def _getDeltaTreeType(cls):
# 		if not cls.deltaTreeType:
# 			from tree import Tree
# 			cls.deltaTreeType = Tree
# 		return cls.deltaTreeType
#
# 	def _setup(self):
# 		self._getDeltaTreeType()
#
# 	def getStructureState(self, interface:TreeInterface)->TreeInterface:
# 		"""returns a separate snapshot of tree at this moment
# 		probably slow but that's ok
# 		we have a janky context block to prevent messing up
# 		global uid register when trees are copied
# 		"""
# 		assert interface
# 		#with interface.ignoreUidChangesInBlock():
# 		state = interface.copy(copyUid=True, useDeepcopy=False, toType=self._getDeltaTreeType())
# 		#print("found state", [i.name for i in state.allBranches()])
# 		return state
#
# 	@classmethod
# 	def extractDeltas(cls,
# 	                  startState:TreeInterface,
# 	                  endState:TreeInterface,
# 	                  targetObject:TreeInterface=None) ->T.List[TreeDeltaAtom]:
# 		"""compare branches between given states"""
# 		#raise
# 		deltas = compareTrees(startState, endState)
# 		#print("extracted deltas")
# 		#pprint.pprint(deltas)
# 		return deltas
#
# 	def onDeltasFound(self, deltas:list[TreeDeltaAtom]):
# 		"""for each tree found changed, send an event to that tree"""
# 		#print("on deltas found", deltas)
# 		changedTrees : set[TreeInterface] = set()
# 		event = TreeDeltaEvent(data=deltas, sender=self	)
#
# 		for i in deltas:
# 			toSync = branchToSyncForDelta(i, self.vendor)
# 			if toSync is None:
# 				continue
# 			event.sender = toSync
# 			#toSync.sendEvent(event)
#
# 		return
#
# 		# had to micromanage events here because performance was pathetic,
# 		# optimise better later (somehow)
#
# 		# get chain of trees to pass event through
# 		allListeners : set[TreeInterface] = set(flatten([i._cachedHierarchyData.nextEventListeners for i in changedTrees]))
#
# 		for i in allListeners:
# 			#print("dispatch event", event.accepted, event)
# 			#i.sendEvent(event)
# 			#i._relayEvent(event, targetChain=)
# 			i._handleEvent(event)
# 			pass
#
# 		#self.vendor.sendEvent(event)
#
#
# 	def onStateChanged(self, changeObject:TreeInterface):
# 		changeObject.sendEvent(changeObject)



