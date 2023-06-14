
from __future__ import annotations

import typing as T
from dataclasses import dataclass

from tree import Tree, Signal

from wp.option import OptionItem, optionType, optionMapFromOptions, FilteredOptionsData, optionItemsFromOptions, optionKeyFromValue, optionFromKey, optionFilterTemplateFn, filterOptions, optionFilterFnType

"""test for a new way of building tools - using a tree structure,
addressing signals and events by strings.

Here we define a typed section for entering information


inspired partially by blueprints, partially by general despair
"""


@dataclass
class TreeFieldParams:
	"""defining auxiliary data, validation, valid options
	etc for tree field

	populate attributes you need, we'll figure it out from there
	"""
	options: optionType = None
	optionFilterFn : optionFilterFnType = None

class TreeField(Tree):
	"""
	Add new signal to check if change of state should propagate
	to other events -
	when updating field to a new value, sometimes just want to update
	the value without triggering a chain of events.

	Use Params object to flag which tree branches to process
	in which way

	"""

	options = None

	def __init__(self, name:str, value=None, uid=None,
	             params:TreeFieldParams=None):
		super(TreeField, self).__init__(name, value, uid)

		self.valueChanged = Signal()
		self.valueChangedPropagate = Signal()
		self.params = params or TreeFieldParams()


	def setValuePropagate(self, value):
		"""set value, and trigger propagate signal"""
		self.setValue(value)
		self.valueChangedPropagate.emit(value)


