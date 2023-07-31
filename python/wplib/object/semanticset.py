
from __future__ import annotations

"""coming back around to try some tagset-like things, but starting off smaller

here is a test for a way of doing set and match operations on multiple elements
within a single line of text :

" (a and b) and not c "
results in "all elements matching a and b, and not matching c" 

"""


class SemanticSet(set):
	"""
	" (a and b) and not c "
	results in "all elements matching a and b, and not matching c"
	"""

	def __init__(self, *args, **kwargs):
		super(SemanticSet, self).__init__(*args, **kwargs)
		self._inverted = False

	def __and__(self, other):
		return type(self)(self.intersection(other))

	def __rand__(self, other):
		return type(self)(self.intersection(other))

	def __or__(self, other):
		return type(self)(self.union(other))

	def __ror__(self, other):
		return type(self)(self.union(other))

	def __not__(self):
		"""flag this set as inverted for further operations"""
		newSet = type(self)(self)
		newSet._inverted = True
		return newSet


if __name__ == '__main__':
	setA = SemanticSet({"a", "b", "c"})
	setB = SemanticSet({"b", "d"})

	print(setA and setB)
	print(setA.__and__(setB))
	print(setA or setB)




