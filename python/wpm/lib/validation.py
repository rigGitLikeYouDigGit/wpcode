from __future__ import annotations
import typing as T

import re

from wplib import string as libstring

from wplib.validation import Rule, RuleSet, ValidationError

from wpm import cmds

"""Maya-specific validation rules"""


class NodeNameUniqueRule(Rule):
	"""check that input resolves to unique node name"""

	def checkInput(self, data):
		if cmds.objExists(data):
			raise ValidationError(f"Node with name {data} already exists")

	def getSuggestedValue(self, data) ->T.Any:
		"""suggest a new name, based on the input"""
		return libstring.incrementName(data)


class NodeNameValidCharactersRule(Rule):
	"""check that input does not contain invalid characters"""
	invalidPattern = re.compile(r"[^a-zA-Z0-9_|:]")

	def checkInput(self, data: str) -> bool:
		if self.invalidPattern.findall(data):
			raise ValidationError(f"Node name {data} contains invalid characters:  {self.invalidPattern.findall(data)}")
		return True

	def getSuggestedValue(self, data) -> T.Any:
		"""replace invalid characters with underscores,
		as Maya does"""
		return self.invalidPattern.sub("_", data)

nodeNameRuleSet = RuleSet(
	[NodeNameUniqueRule(), NodeNameValidCharactersRule()]
)


