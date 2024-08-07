

from __future__ import annotations
import typing as T

from dataclasses import dataclass

from wplib import log
from wplib.sequence import flatten
from wplib.string import multiSplit, indicesOfAny, splitIndices, mergeRepeated

"""base for any object class that  
can be traversed through path sequences - 
for example a tree.

Path may be a sequence of strings, or a connected string

"""

# test a more extensible and explicit way of packing arguments
@dataclass
class TraversableParams:
	pass


class Traversable:
	"""
	some systems may define different "directions" through strings -
	consider hierarchical nodes with attributes.
	my/node.attribute
	my/leaf/node.attribute
	one direction is assumed as default

	consider, should the traversing be modal? That is, the following would be valid:
	my/node.attributeRoot/attributeBranch

	I guess that would be up to the attributeRoot object to define its path separators,
	since we only solve one part at a time.

	split address above into ("my", "/", "node", ".", "attributeRoot", "/", "attributeBranch")
	Only work on first tokens, then remove them when resolved, to pass to children

	in the general case, this object is a node, with different types of edges to other objects.


	stretch goal: slicing and fuzzy matching

	"""
	TraversableParams = TraversableParams

	# maybe put this in an enum or namespace, but it's also fine like this
	separatorChars = {
		"child" : "/",
		"attribute" : ".",
		#"parent" : "^",
	}

	wildCardChars = {}

	def defaultStartSeparator(self)->str:
		"""return default start separator to use if none is given"""
		return "/"

	@classmethod
	def defaultTraverseParamCls(cls)->T.Type[TraversableParams]:
		return TraversableParams

	@classmethod
	def checkTraversalDestNameValid(cls, destName:str)->bool:
		"""check that given destination name is valid for traversal - basically don't allow adding
		names with traversal characters in them
		"""
		return not any(c in destName for c in cls.separatorChars.values())

	"""logic when given a path - 
	flatten top level, and check first letter of first token
	- if that token is a single sepchar, use it and the next token
	- if it begins with a sepchar, use it
	- otherwise use default sepchar
	"""

	def _getCharAndFirstTokenAndBody(self, path:(str, tuple[str]))->(tuple[str, str, str],
			tuple[None, None, None]):
		"""return (separator char, first token, rest of path body)
		only look at first occurrence
		this is a stupid system that probably doesn't handle
		every edge case between sep chars yet
		"""

		sepChar = ""
		firstToken = ""
		body = ""

		tokens = list(flatten([path]))
		#print("tokens", tokens)
		if not tokens:
			return None, None, None

		firstChunk : str = tokens.pop(0)

		if firstChunk in self.separatorChars.values():
			# use first token as separator
			sepChar = firstChunk
			#firstChunk = self.defaultStartSeparator() + tokens.pop(0)
			firstChunk = tokens.pop(0)
		elif firstChunk[0] in self.separatorChars.values():
			sepChar = firstChunk[0]
			firstChunk = firstChunk[1:]
		else:
			firstChunk = firstChunk
			sepChar = self.defaultStartSeparator()

		while firstChunk[0] in self.separatorChars.values():
			firstChunk = firstChunk[1:]

		#print(sepChar, firstChunk, tokens)
		# firstToken, chunk = multiSplit(firstChunk, self.separatorChars.values(), preserveSepChars=True)
		#firstChunk = mergeRepeated(firstChunk, self.separatorChars.values())
		toSplitIndices = indicesOfAny(firstChunk, self.separatorChars.values())
		#print("toSplitIndices", toSplitIndices)
		toSplitIndices.append(len(firstChunk))
		firstToken = firstChunk[:toSplitIndices[0]]

		# separate rest of chunk into its own block, assume it starts with a
		# separator char if it's not empty

		#print("firstChunk", firstChunk, "toSplitIndices", toSplitIndices)
		chunkBody = firstChunk[toSplitIndices[0]:]
		if chunkBody:
			tokens.insert(0, chunkBody)

		assert sepChar
		assert firstToken
		return sepChar, firstToken, tokens


	def splitTraversalPath(self, path:(str, tuple[str]))->list[str, ...]:
		"""split traversal path input into tuple of strings -
		guarantee (sepChar, name, sepChar, name) etc
		"""
		finalTokens = []
		for i in flatten(path):
			indices = indicesOfAny(i, self.separatorChars.values())
			# split before and after each index
			indices = [0, *indices, len(i)]
			finalTokens.extend(splitIndices(i, indices))
		return finalTokens

	def sanitiseRawPath(self, path:(str, tuple[str])):
		"""return a path with first token guaranteed to start with
		a separator char (or the detault)"""
		pathTokens = flatten([path,])
		if not pathTokens:
			return ()
		if not path[0] in self.separatorChars.values():
			pathTokens[0] = self.defaultStartSeparator() + pathTokens[0]
		return tuple(pathTokens)

		#return tuple(self.splitTraversalPath(path))

	keyT = (str, tuple[str, ...])
	def parseFirstToken(self, address:keyT)->tuple[str, ...]:
		"""parse an address and retrieve its first token.
		We assume that the first token will not just be a separator char"""
		firstTokens = multiSplit(
			address[0], self.separatorChars.values(),
			preserveSepChars=True)
		#print("first tokens", firstTokens)


		return firstTokens[0], "".join(firstTokens[1:])

	def findNextTraversable(self, separator:str, token:str,
	                        params:defaultTraverseParamCls())->Traversable:
		"""find next object from separator and token.
		In the Tree, this will also take care of branch creation on lookup -
		I don't have a good way to separate them, and they're both tied closely
		enough to the specific Tree implementation, that calling them both
		here feels ok

		:param separator: separator "direction" character, the default character if not specified.
		:param token: single token to look up.
		:param params: params object to use during lookup - should not be modified in place, as it will be passed to children

		:raises LookupError: if path cannot be resolved
		"""
		raise NotImplementedError

	def buildTraverseParamsFromRawKwargs(self, **kwargs)->TraversableParams:
		"""build params object from raw kwargs"""
		return self.defaultTraverseParamCls()(**kwargs)

	def traverse(self, path, traverseParams:(defaultTraverseParamCls(), None), **kwargs)->Traversable:
		"""all paths are assumed relative from this object -
		a leading separator is used for direction, with the default used
		if missing.

		we require explicitly constructed params, or None

		:raises LookupError: if path cannot be resolved
		"""

		if not path: # empty path, return self
			return self
		#print("flattened path", path)
		sepChar, firstToken, body = self._getCharAndFirstTokenAndBody(
			path	)
		if sepChar is None: # path was an empty list of lists?
			return self
		#print("sep, first, body:", sepChar, firstToken, body)

		if traverseParams is None:
			traverseParams = self.buildTraverseParamsFromRawKwargs(**kwargs)

		#log(sepChar, firstToken)
		# look up target from separator and token
		target = self.findNextTraversable(sepChar, firstToken, params=traverseParams)


		#return target(body, **kwargs)
		return target.traverse(body, traverseParams, **kwargs)

	"""
	recommend copying this code for whatever entrypoint
	you use for traversal
	not worth defining it here
	"""

	# def __call__(self, *path, traverseParams:defaultTraverseParamCls()=None, **kwargs)->Traversable:
	# 	"""syntax sugar for traverse"""
	#
	# 	path = flatten(path)
	# 	if traverseParams is None:
	# 		traverseParams = self.buildTraverseParamsFromRawKwargs(**kwargs)
	# 	return self.traverse(path, traverseParams, **kwargs)


