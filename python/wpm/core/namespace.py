from __future__ import annotations

"""functions helping with maya namespaces"""

from tree import Tree, TreeType, TreeInterface


from .cache import om, cmds, getCache

def _validate(s):
	return om.MNamespace.validateName(s)

class NamespaceTree(TreeInterface):
	"""tree modelling maya's global scene namespace"""

	separatorChar = ":"

	@classmethod
	def defaultBranchCls(cls):
		return cls

	def __init__(self, name=":", *args, **kwargs):
		super(NamespaceTree, self).__init__(name)

	def stringAddress(self, includeRoot=False) -> str:
		return om.MNamespace.validateName(":" + ":".join(self.address(includeSelf=True))).strip() or ":"

	def __repr__(self):
		return "<{} ({}) : {}>".format(self.__class__, self.name, self._value)

	@staticmethod
	def _nsName(ns):
		return next(filter(None, om.MNamespace.validateName(ns).split(":")))

	@property
	def branches(self) -> list[TreeType]:

		branches = []
		for i in om.MNamespace.getNamespaces(	self.stringAddress()):
			branches.append(NamespaceTree(self._nsName(i)))
		return branches

	def _createChildBranch(self, name, kwargs) ->TreeType:
		"""manually create new ns"""
		newName = ":" + self._nsName(self.stringAddress() + ":" + name)
		if not om.MNamespace.namespaceExists(newName):
			om.MNamespace.addNamespace(newName)

		baseResult = super(NamespaceTree, self)._createChildBranch(name, kwargs)
		print("baseResult", baseResult)
		return baseResult

	def objects(self)->tuple[om.MObject]:
		return om.MNamespace.getNamespaceObjects(self.stringAddress())

	def addNode(self, node:om.MObject):
		om.MFnDependencyNode(node).setName(
			self.stringAddress() + ":" + om.MNamespace.getNamespaceFromName(node)
		)

	def ensureExists(self):
		if self.parent:
			self.parent.ensureExists()
		if not om.MNamespace.namespaceExists(self.stringAddress()):
			om.MNamespace.addNamespace(self.stringAddress())

	def remove(self, address:(keyType, TreeInterface, None)):
		om.MNamespace.removeNamespace(self.stringAddress())




def getNamespaceTree()->NamespaceTree:
	return NamespaceTree(":")

