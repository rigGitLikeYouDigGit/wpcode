

from __future__ import annotations
import typing as T

from dataclasses import dataclass

from wplib.sequence import flatten
from wplib.string import multiSplit

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
	"""some systems may define different "directions" through strings -
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


	"""
	TraversableParams = TraversableParams

	# maybe put this in an enum or namespace, but it's also fine like this
	separatorChars = {
		"/" : "hierarchy",
		#"." : "attribute"
	}

	def defaultStartSeparator(self)->str:
		"""return default start separator to use if none is given"""
		return "/"

	@classmethod
	def defaultTraverseParamCls(cls)->T.Type[TraversableParams]:
		return TraversableParams


	keyT = (str, tuple[str, ...])

	def parseFirstToken(self, address:keyT)->tuple[str, str]:
		"""parse an address and retrieve its first token.
		We assume that the first token will not just be a separator char"""
		firstTokens = multiSplit(
			address[0], self.separatorChars.keys(), preserveSepChars=True)
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
		path = flatten(path)
		#print("traverse", path, kwargs)
		if not path: # empty path, return self
			return self
		#print("flattened path", path)

		if traverseParams is None:
			traverseParams = self.buildTraverseParamsFromRawKwargs(**kwargs)

		# first parse address into tokens
		first, body = self.parseFirstToken(path)
		if first in self.separatorChars:
			# use first token as separator
			separator = first
			first, body = self.parseFirstToken(body)
		else:
			# use default separator
			separator = self.defaultStartSeparator()

		#print("traverse", path, first, body, separator)

		body = ((body,) if body else ()) + tuple(path[1:])


		# look up target from separator and token
		target = self.findNextTraversable(separator, first, params=traverseParams)


		#return target(body, **kwargs)
		return target.traverse(body, traverseParams, **kwargs)

	def __call__(self, *path, traverseParams:defaultTraverseParamCls()=None, **kwargs)->Traversable:
		"""syntax sugar for traverse"""

		path = flatten(path)
		#print("call base", path)
		if traverseParams is None:
			traverseParams = self.buildTraverseParamsFromRawKwargs(**kwargs)
		return self.traverse(path, traverseParams, **kwargs)


