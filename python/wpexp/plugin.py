
from __future__ import annotations
import typing as T

"""
another test at a plugin-based system, letting later
uses specialise the tokens used,
letting users add their own tokens, etc


"""



class ExpWarning(Exception):
	"""raise a warning that the given token
	WOULD cause a hard error if it were to be evaluated

	by convention, call with
	ExpWarning(token, message)
	"""

class ExpError(Exception):
	"""raise a hard error when a token cannot be resolved"""


class SceneExpPlugin:
	"""example of common tokens to use across
	most 3d dccs """
	addTokens = [
		"SCENE_DIR",
		"SCENE_NAME"
	]

	def resolveToken(self, token:str)->str:
		"""resolve a token to a value"""



