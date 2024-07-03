
from __future__ import annotations
import typing as T

import fnmatch
import whoosh as wh

SPECIAL_CHARS = ('*', '?', '[', ']')

def definesMatch(query:str)->bool:
	"""does this query define a match"""
	return bool(set(query) & set(SPECIAL_CHARS))

def getMatches(
		items: T.Iterable[dict[str, T.Any]],
		query: dict[str, str],
)->T.List[dict[str, T.Any]]:
	results = []
	for item in items:
		if all(
				fnmatch.fnmatch(str(item.get(k, '')), v)
				for k, v in query.items()
		):
			results.append(item)

	return results



	pass
