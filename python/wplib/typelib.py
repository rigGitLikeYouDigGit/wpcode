

def isImmutable(obj):
	return isinstance(obj, (int, float, str, bool, tuple, frozenset, type(None)))

def isSeq(obj):
	return isinstance(obj, (list, tuple, set, frozenset))
